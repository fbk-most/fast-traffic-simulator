"""fast-traffic-simulator: NumPy/SciPy-based microscopic traffic simulator.

This module contains the core :class:`Simulator` class that models individual
vehicles moving through a directed road network represented as a weighted graph.
Each simulation step advances all vehicles by one time unit simultaneously.

Network model
-------------
The road network is a directed graph in which **edges** (links) represent road
segments and **nodes** represent intersections or junctions.  Node indices must
be contiguous non-negative integers; edge indices are assigned implicitly by
the row order of the edges DataFrame.

Routing
-------
Routes are computed using **Dijkstra's shortest-path algorithm** with travel
time (edge length divided by free-flow speed) as the edge weight.  The
algorithm is run once at :meth:`Simulator.build` time over the full network,
producing a predecessor matrix that encodes the next hop for every
origin–destination pair.

Routes can be refreshed periodically during the simulation by calling
``step(update_next_leg=True)``.  In that case, instantaneous measured speeds
(derived from vehicle displacements in the current step) are substituted for
free-flow speeds, so the updated paths reflect current congestion.

Vehicle-following model (links)
--------------------------------
Vehicle dynamics on links follow **Newell's simplified car-following model**.
Each vehicle travels as fast as possible subject to two constraints:

1. It may not exceed the free-flow speed of its current edge.
2. It must maintain a minimum safe spacing of ``DELTA = 1 / MAX_DENSITY``
   units behind the vehicle immediately ahead in the same lane.

This model naturally reproduces stop-and-go waves, queue formation, and
congestion without requiring explicit capacity constraints on edges.

Node model
----------
Nodes are treated as **dimensionless points**: travel time accrues only on
links, and the transfer of a vehicle from an incoming edge to an outgoing edge
is instantaneous.  In each time step, vehicles that have reached the end of an
incoming edge and whose next edge (according to the current routing) has
available inbound capacity are immediately transferred to that edge.  Vehicles
that reach their destination node are removed from the network.

Lane model
----------
Each edge may have one or more lanes.  Lanes are **parallel and independent**:
vehicles in different lanes of the same edge do not interact, and no lane
changes occur.  When a vehicle enters an edge it is assigned to the lane with
the most available space (largest gap to the current rear vehicle); it remains
in that lane until it exits the edge.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import dijkstra as shortest_path_search


class Simulator:
    """Microscopic traffic simulator for directed road networks.

    The simulator represents the road network as a directed graph where edges
    correspond to road segments and nodes correspond to intersections. Vehicles
    travel from an origin node to a destination node following the shortest path
    (by travel time). Multiple lanes per edge are supported, and vehicles maintain
    a safe following distance of at least ``DELTA`` units.

    Attributes:
        MAX_DENSITY: Maximum vehicle density per unit length (vehicles per metre).
        DELTA: Minimum safe following distance (reciprocal of MAX_DENSITY).
        END_OF_TIME: Sentinel value used for vehicles that have not yet arrived.
    """

    MAX_DENSITY: np.float32 = np.float32(0.2)
    DELTA: np.float32 = np.float32(1.0 / MAX_DENSITY)
    END_OF_TIME: int = 999_999

    class VehicleStatus(Enum):
        """Lifecycle states of a vehicle.

        Attributes:
            WAITING: Vehicle has not yet departed (start time not reached).
            AT_NODE: Vehicle is at an intersection, about to enter an edge.
            IN_EDGE: Vehicle is travelling along a road segment.
            ARRIVED: Vehicle has reached its destination.
        """

        WAITING = auto()
        AT_NODE = auto()
        IN_EDGE = auto()
        ARRIVED = auto()

    @dataclass
    class VehiclesRecord:
        """Per-vehicle state arrays.

        All fields that are not provided at construction time are initialised
        in :meth:`__post_init__`.

        Args:
            origin: Integer node index of each vehicle's origin.
            destination: Integer node index of each vehicle's destination.
            start_time: Simulation time step at which each vehicle departs.

        Attributes:
            status: Current :class:`VehicleStatus` value (stored as ``int32``).
            node: Current node index for each vehicle.
            edge: Current edge index (``-1`` when not in an edge).
            lane: Current lane index within the edge (``-1`` when not in an edge).
            edge_distance: Distance travelled along the current edge (metres).
            lane_front_vehicle: Index of the vehicle immediately ahead in the
                same lane (``-1`` if none).
            arrival_time: Simulation step at which the vehicle arrived
                (``END_OF_TIME`` if not yet arrived).
            started_now: Boolean mask — ``True`` for vehicles that departed this
                step.
            arrived_now: Boolean mask — ``True`` for vehicles that arrived this
                step.
            entered_edge_now: Boolean mask — ``True`` for vehicles that entered
                an edge this step.
        """

        origin: np.ndarray
        destination: np.ndarray
        start_time: np.ndarray
        status: np.ndarray = field(init=False)
        node: np.ndarray = field(init=False)
        edge: np.ndarray = field(init=False)
        lane: np.ndarray = field(init=False)
        edge_distance: np.ndarray = field(init=False)
        lane_front_vehicle: np.ndarray = field(init=False)
        arrival_time: np.ndarray = field(init=False)
        started_now: np.ndarray = field(init=False)
        arrived_now: np.ndarray = field(init=False)
        entered_edge_now: np.ndarray = field(init=False)

        def __post_init__(self) -> None:
            nr_vehicles = self.origin.shape[0]
            self.status = np.full(nr_vehicles, Simulator.VehicleStatus.WAITING.value, dtype=np.int32)
            self.node = self.origin.copy()
            self.edge = np.full(nr_vehicles, -1, dtype=np.int32)
            self.lane = np.full(nr_vehicles, -1, dtype=np.int32)
            self.lane_front_vehicle = np.full(nr_vehicles, -1, dtype=np.int32)
            self.edge_distance = np.full(nr_vehicles, 0.0, dtype=np.float32)
            self.arrival_time = np.full(nr_vehicles, Simulator.END_OF_TIME, dtype=np.int32)
            self.started_now = np.full(nr_vehicles, False, dtype=np.bool_)
            self.arrived_now = np.full(nr_vehicles, False, dtype=np.bool_)
            self.entered_edge_now = np.full(nr_vehicles, False, dtype=np.bool_)

    @dataclass
    class EdgesRecord:
        """Per-edge state arrays.

        Args:
            from_node: Origin node index for each edge.
            to_node: Destination node index for each edge.
            length: Length of each edge in metres.
            speed: Free-flow speed of each edge (metres per time step).
            lanes: Number of lanes on each edge.
            nr_vehicles: Cumulative vehicle count used for traffic-volume logging.

        Attributes:
            last_vehicle: Shape ``(nr_edges, max_lanes)`` array storing the index
                of the rearmost vehicle in each lane. ``-1`` means the lane is
                empty; ``-2`` means the lane does not exist for that edge.
        """

        from_node: np.ndarray
        to_node: np.ndarray
        length: np.ndarray
        speed: np.ndarray
        lanes: np.ndarray
        nr_vehicles: np.ndarray
        last_vehicle: np.ndarray = field(init=False)

        def __post_init__(self) -> None:
            nr_edges = self.from_node.shape[0]
            self.last_vehicle = np.empty((nr_edges, self.lanes.max()), dtype=np.int32)
            for lane in range(self.lanes.max()):
                # -1: lane exists but is empty; -2: lane does not exist for this edge
                self.last_vehicle[:, lane] = np.where(self.lanes > lane, -1, -2)

    def __init__(
        self,
        edges,
        vehicles,
        *,
        random: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Store edge and vehicle data without computing routes.

        This constructor is intentionally lightweight. Call :meth:`build`
        instead to obtain a fully initialised, ready-to-run simulator.

        Args:
            edges: DataFrame with columns ``from``, ``to``, ``length``,
                ``speed``, and ``lanes``. Node indices must be contiguous
                integers starting from 0.
            vehicles: DataFrame with columns ``origin``, ``destination``, and
                ``start`` (departure time step).
            random: If ``True``, vehicles competing for the same edge are
                shuffled randomly before priority is assigned. Defaults to
                ``False`` (deterministic, first-in-first-served).
            rng: NumPy random Generator to use when ``random=True``. If
                ``None`` and ``random=True``, a fresh Generator is created via
                ``numpy.random.default_rng()``.
        """
        self._random = random
        self._rng: np.random.Generator | None = (
            np.random.default_rng() if (random and rng is None) else rng
        )
        self._nr_vehicles = vehicles.shape[0]
        self._vehicles = Simulator.VehiclesRecord(
            origin=vehicles['origin'].values.astype(np.int32),
            destination=vehicles['destination'].values.astype(np.int32),
            start_time=vehicles['start'].values.astype(np.int32),
        )

        self._nr_edges = edges.shape[0]
        self._edges = Simulator.EdgesRecord(
            from_node=edges['from'].values.astype(np.int32),
            to_node=edges['to'].values.astype(np.int32),
            length=edges['length'].values.astype(np.float32),
            speed=edges['speed'].values.astype(np.float32),
            lanes=edges['lanes'].values.astype(np.int32),
            nr_vehicles=np.zeros(self._nr_edges, np.int32),
        )

        self._max_lanes = self._edges.lanes.max()

        n_nodes = int(max(edges['from'].max(), edges['to'].max())) + 1
        self._nodes_to_edge_map = coo_array(
            (range(self._nr_edges), (edges['from'], edges['to'])),
            shape=(n_nodes, n_nodes),
        ).toarray()

        self._next_leg: np.ndarray | None = None
        self._ready = False
        self._now = 0

    @classmethod
    def build(
        cls,
        edges,
        vehicles,
        *,
        fix_unreachable: bool = False,
        random: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple['Simulator', list[int]]:
        """Construct a fully initialised simulator ready to run.

        This is the preferred entry point. It performs the expensive
        shortest-path computation and validates (or fixes) vehicle
        destinations before the simulation loop starts.

        Args:
            edges: DataFrame with columns ``from``, ``to``, ``length``,
                ``speed``, and ``lanes``. Node indices must be contiguous
                integers starting from 0.
            vehicles: DataFrame with columns ``origin``, ``destination``, and
                ``start`` (departure time step).
            fix_unreachable: Controls behaviour when vehicles with unreachable
                destinations are found (i.e. no path exists from origin to
                destination). If ``True``, their destination is reset to their
                origin so they depart and immediately arrive; the list of
                affected vehicle indices is returned. If ``False`` (default),
                a :exc:`ValueError` is raised instead.
            random: If ``True``, vehicles competing for the same edge are
                shuffled randomly before priority is assigned. Defaults to
                ``False`` (deterministic, first-in-first-served).
            rng: NumPy random Generator forwarded to the simulator when
                ``random=True``. Pass the same Generator used for
                replica-level stochasticity to make runs fully reproducible.
                If ``None`` and ``random=True``, a fresh Generator is created.

        Returns:
            A two-tuple ``(simulator, fixed)`` where *simulator* is ready to
            call :meth:`step` on, and *fixed* is the list of vehicle indices
            whose destination was reset to their origin (empty when nothing
            needed fixing).

        Raises:
            ValueError: If ``fix_unreachable=False`` and one or more vehicles
                have unreachable destinations.
        """
        simulator = cls(edges, vehicles, random=random, rng=rng)

        # Compute the routing.
        # Note that from/to are inverted so that Dijkstra returns the *successor*
        # node on the path from each source to each destination.
        n_nodes = simulator._nodes_to_edge_map.shape[0]
        graph = coo_array(
            (edges['length'] / edges['speed'], (edges['to'], edges['from'])),
            shape=(n_nodes, n_nodes),
        ).toarray()
        _, simulator._next_leg = shortest_path_search(graph, return_predecessors=True)

        # Check for vehicles with unreachable destinations
        unreachable_mask = (
            (simulator._vehicles.destination != simulator._vehicles.origin) &
            (simulator._next_leg[
                simulator._vehicles.destination,
                simulator._vehicles.origin,
            ] == -9999)
        )
        fixed: list[int] = []
        if unreachable_mask.any():
            if not fix_unreachable:
                raise ValueError(
                    f"{unreachable_mask.sum()} vehicle(s) have unreachable destinations. "
                    "Pass fix_unreachable=True to reroute them to their origin."
                )
            simulator._vehicles.destination[unreachable_mask] = (
                simulator._vehicles.origin[unreachable_mask]
            )
            fixed = unreachable_mask.nonzero()[0].tolist()

        simulator._ready = True
        return simulator, fixed

    @property
    def vehicles(self) -> 'Simulator.VehiclesRecord':
        """Per-vehicle state arrays.

        Returns the live :class:`VehiclesRecord` instance. Individual array
        fields may be read or modified in place; do not replace the record
        object itself.
        """
        return self._vehicles

    @property
    def edges(self) -> 'Simulator.EdgesRecord':
        """Per-edge state arrays.

        Returns the live :class:`EdgesRecord` instance. Individual array
        fields may be read or modified in place; do not replace the record
        object itself.
        """
        return self._edges

    def step(self, update_next_leg: bool = False) -> None:
        """Advance the simulation by one time step.

        The method updates vehicle states in the following order:

        1. **Starting** — vehicles whose departure time equals the current step
           transition from ``WAITING`` to ``AT_NODE``.
        2. **Out-of-edge** — vehicles that have travelled the full length of
           their edge transition from ``IN_EDGE`` to ``AT_NODE``.
        3. **Arrived** — vehicles at their destination node transition to
           ``ARRIVED``.
        4. **Entering edge** — vehicles at a node attempt to enter the next
           edge on their shortest path. The loop repeats until no more
           vehicles can enter (handles multi-hop moves within one step).
        5. **Progress** — vehicles in edges advance by their edge speed, up
           to the front vehicle's position minus ``DELTA``.

        Args:
            update_next_leg: If ``True``, recompute shortest paths using
                instantaneous measured speeds after updating vehicle positions.
                This is expensive and is typically done only every few hundred
                steps. Defaults to ``False``.
        """
        if not self._ready:
            raise RuntimeError(
                "Simulator.build() must be called before step(). "
                "Use Simulator.build(edges, vehicles) to create a ready simulator."
            )
        assert self._next_leg is not None
        next_leg: np.ndarray = self._next_leg

        waiting_status = Simulator.VehicleStatus.WAITING.value
        at_node_status = Simulator.VehicleStatus.AT_NODE.value
        in_edge_status = Simulator.VehicleStatus.IN_EDGE.value
        arrived_status = self.VehicleStatus.ARRIVED.value

        self._vehicles.started_now[:] = False
        self._vehicles.arrived_now[:] = False
        self._vehicles.entered_edge_now[:] = False

        def do_starting() -> None:
            # Find vehicles that are ready to start
            starting = (self._vehicles.start_time == self._now) & (self._vehicles.status == waiting_status)
            if np.any(starting):
                self._vehicles.status[starting] = at_node_status
                self._vehicles.started_now[starting] = True

        do_starting()

        def do_out_of_edge() -> None:
            # Find vehicles that have reached the end of their edge
            vehicles = (
                (self._vehicles.status == in_edge_status) &
                (self._vehicles.edge_distance == self._edges.length[self._vehicles.edge])
            )
            self._vehicles.status[vehicles] = at_node_status

        do_out_of_edge()

        def do_arrived() -> None:
            # Find vehicles that reached their final destination
            arrived = (
                (self._vehicles.status == at_node_status) &
                (self._vehicles.node == self._vehicles.destination)
            )
            if np.any(arrived):
                self._vehicles.status[arrived] = arrived_status
                self._vehicles.arrival_time[arrived] = self._now
                self._vehicles.arrived_now[arrived] = True

                # Restrict to vehicles that did not start right away (origin == destination)
                arrived_not_started = arrived & (self._vehicles.edge >= 0)
                # Restrict to vehicles that are last in their lane
                in_out_vehicles = (
                    self._edges.last_vehicle[
                        self._vehicles.edge[arrived_not_started],
                        self._vehicles.lane[arrived_not_started],
                    ] == arrived_not_started.nonzero()[0]
                )
                in_out_vehicles = arrived_not_started.nonzero()[0][in_out_vehicles]
                # Mark those lanes as empty
                self._edges.last_vehicle[
                    self._vehicles.edge[in_out_vehicles],
                    self._vehicles.lane[in_out_vehicles],
                ] = -1
                # Reset edge and lane for arrived vehicles
                self._vehicles.edge[arrived_not_started] = -1
                self._vehicles.lane[arrived_not_started] = -1

        do_arrived()

        def do_entering_edge() -> None:
            while True:
                # Find vehicles that should enter an edge
                entering_edge = (self._vehicles.status == at_node_status)
                if not np.any(entering_edge):
                    break
                # Compute the next node and edge.
                # Note that from/to are inverted in _next_leg.
                next_nodes = next_leg[
                    self._vehicles.destination[entering_edge],
                    self._vehicles.node[entering_edge],
                ]
                next_edges = self._nodes_to_edge_map[
                    self._vehicles.node[entering_edge], next_nodes
                ]

                assert not (next_nodes == -9999).any()

                # Get unique edges targeted by the vehicles, and the corresponding vehicles.
                # Shuffling is performed for random simulations.
                if self._random:
                    assert self._rng is not None
                    order = np.arange(len(next_edges))
                    self._rng.shuffle(order)
                    unique_edges, unique_vehicles = np.unique(next_edges[order], return_index=True)
                    unique_vehicles = order[unique_vehicles]
                else:
                    unique_edges, unique_vehicles = np.unique(next_edges, return_index=True)

                # Restrict to edges that have at least one free lane
                lane_distances = np.select(
                    [
                        self._edges.last_vehicle[unique_edges] == -1,
                        self._edges.last_vehicle[unique_edges] < -1,
                        True,
                    ],
                    [
                        999_999.9,
                        0.0,
                        self._vehicles.edge_distance[self._edges.last_vehicle[unique_edges]],
                    ],
                )
                free_edges = (lane_distances >= Simulator.DELTA).any(axis=1)

                if not free_edges.any():
                    break

                edges = unique_edges[free_edges]
                lanes = np.argmax(lane_distances[free_edges], axis=1)
                vehicles = entering_edge.nonzero()[0][unique_vehicles[free_edges]]
                assert not (self._edges.last_vehicle[edges, lanes] < -1).any()

                # If a vehicle was last in its previous lane, mark that lane as empty
                in_out_vehicles = (
                    (self._vehicles.edge[vehicles] != -1) &
                    (self._edges.last_vehicle[
                        self._vehicles.edge[vehicles],
                        self._vehicles.lane[vehicles],
                    ] == vehicles)
                )
                self._edges.last_vehicle[
                    self._vehicles.edge[vehicles[in_out_vehicles]],
                    self._vehicles.lane[vehicles[in_out_vehicles]],
                ] = -1

                # Transition to IN_EDGE and update all auxiliary fields
                self._vehicles.status[vehicles] = in_edge_status
                self._vehicles.entered_edge_now[vehicles] = True
                self._vehicles.edge[vehicles] = edges
                self._vehicles.lane[vehicles] = lanes
                self._vehicles.node[vehicles] = next_nodes[unique_vehicles[free_edges]]
                self._vehicles.edge_distance[vehicles] = 0.0
                self._vehicles.lane_front_vehicle[vehicles] = self._edges.last_vehicle[edges, lanes]
                self._edges.last_vehicle[edges, lanes] = vehicles
                self._edges.nr_vehicles[edges] += 1

        do_entering_edge()

        def do_progress_vehicles() -> None:
            # Clear stale front-vehicle references (front vehicle has changed edge)
            new_edge_front = (
                (self._vehicles.lane_front_vehicle != -1) &
                (self._vehicles.edge != self._vehicles.edge[self._vehicles.lane_front_vehicle])
            )
            self._vehicles.lane_front_vehicle[new_edge_front] = -1

            in_edge = (self._vehicles.status == in_edge_status)
            in_edge_front = in_edge & (self._vehicles.lane_front_vehicle == -1)
            in_edge_follow = in_edge & ~in_edge_front

            # Following vehicles advance up to (front vehicle position - DELTA)
            new_distance = np.minimum(
                self._vehicles.edge_distance[in_edge_follow]
                + self._edges.speed[self._vehicles.edge[in_edge_follow]],
                self._vehicles.edge_distance[self._vehicles.lane_front_vehicle[in_edge_follow]]
                - Simulator.DELTA,
            )

            if update_next_leg:
                def do_update_next_leg() -> None:
                    step = new_distance - self._vehicles.edge_distance[in_edge_follow]
                    step_edge = self._vehicles.edge[in_edge_follow]
                    instant_speed = self._edges.speed.copy()
                    instant_speed[step_edge] = 0.01
                    np.add.at(instant_speed, step_edge, step)
                    unique_edge, count_edge = np.unique(step_edge, return_counts=True)
                    np.divide.at(instant_speed, unique_edge, count_edge)
                    graph = coo_array(
                        (self._edges.length / instant_speed, (self._edges.to_node, self._edges.from_node))
                    ).toarray()
                    _, self._next_leg = shortest_path_search(graph, return_predecessors=True)

                do_update_next_leg()

            self._vehicles.edge_distance[in_edge_follow] = new_distance

            # Leading vehicles advance up to the edge length
            self._vehicles.edge_distance[in_edge_front] = np.minimum(
                self._vehicles.edge_distance[in_edge_front]
                + self._edges.speed[self._vehicles.edge[in_edge_front]],
                self._edges.length[self._vehicles.edge[in_edge_front]],
            )

        do_progress_vehicles()

        assert not (self._vehicles.edge_distance < 0.0).any()

        self._now += 1
