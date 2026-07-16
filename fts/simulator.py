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
algorithm is run once at :meth:`Simulator.build` time, from every distinct
vehicle destination, producing a predecessor matrix that encodes the next hop
towards each destination from any node.

Routes can be refreshed periodically during the simulation by calling
``step(update_next_leg=True)``.  In that case, instantaneous measured speeds
(derived from vehicle displacements in the current step) are substituted for
free-flow speeds, so the updated paths reflect current congestion.  When the
caller commits to a maximum refresh interval (the ``horizon`` argument), the
recomputation is restricted to the destinations whose routing can actually be
read before the next refresh, which is considerably cheaper and produces
identical results; a :exc:`RuntimeError` is raised if the promised refresh
does not happen in time.

Refreshes can also be automated and parallelised: building the simulator with
``refresh_interval`` makes :meth:`Simulator.step` trigger the recomputation
itself on a fixed cadence, and ``refresh_workers`` distributes it over a pool
of worker processes (with identical results, since each destination's
shortest-path tree is computed independently).

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
available inbound capacity are immediately transferred to that edge.  Inbound
capacity is evaluated once per step: lane space vacated during a step becomes
available to entering vehicles only in the following step.  Vehicles that
reach their destination node are removed from the network.

Lane model
----------
Each edge may have one or more lanes.  Lanes are **parallel and independent**:
vehicles in different lanes of the same edge do not interact, and no lane
changes occur.  When a vehicle enters an edge it is assigned to the lane with
the most available space (largest gap to the current rear vehicle); it remains
in that lane until it exits the edge.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra as shortest_path_search

# Static network data of the worker processes used for parallel shortest-path
# computation; populated once per worker by _refresh_pool_initializer.
_pool_network: dict = {}


def _refresh_pool_initializer(to_node: np.ndarray, from_node: np.ndarray, n_nodes: int) -> None:
    _pool_network['to_node'] = to_node
    _pool_network['from_node'] = from_node
    _pool_network['n_nodes'] = n_nodes


def _refresh_pool_worker(args: tuple) -> np.ndarray:
    """Compute predecessor rows for a chunk of target nodes."""
    travel_time, targets = args
    n_nodes = _pool_network['n_nodes']
    graph = csr_array(
        (travel_time, (_pool_network['to_node'], _pool_network['from_node'])),
        shape=(n_nodes, n_nodes),
    )
    _, predecessors = shortest_path_search(
        graph, return_predecessors=True, indices=targets
    )
    return predecessors


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
            status: Current :class:`VehicleStatus` value (stored as ``int8``).
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
            self.status = np.full(nr_vehicles, Simulator.VehicleStatus.WAITING.value, dtype=np.int8)
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
        seed: int | np.random.Generator | None = None,
        refresh_interval: int | None = None,
        refresh_workers: int | None = None,
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
            seed: Seed (or ready-made :class:`numpy.random.Generator`) for the
                shuffling performed when ``random=True``. Passing an integer
                makes random runs reproducible. Defaults to ``None``
                (fresh entropy on every run).
            refresh_interval: If set, :meth:`step` triggers the shortest-path
                recomputation automatically every ``refresh_interval`` steps
                (with the matching horizon), so callers no longer need to pass
                ``update_next_leg``/``horizon``. Defaults to ``None`` (caller
                -driven refreshes only).
            refresh_workers: If set to 2 or more, shortest-path recomputations
                are distributed over a persistent pool of that many worker
                processes. Results are identical to the single-process
                computation. Defaults to ``None`` (compute in-process).
        """
        self._random = random
        self._rng = np.random.default_rng(seed)
        if refresh_interval is not None and refresh_interval < 1:
            raise ValueError("refresh_interval must be a positive number of steps")
        self._refresh_interval = refresh_interval
        self._refresh_workers = refresh_workers
        self._refresh_pool: ProcessPoolExecutor | None = None
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

        self._n_nodes = int(max(edges['from'].max(), edges['to'].max())) + 1

        # Sorted-key lookup mapping a (from_node, to_node) pair to its edge
        # index; O(nr_edges) memory instead of a dense nodes x nodes matrix.
        keys = (
            self._edges.from_node.astype(np.int64) * self._n_nodes
            + self._edges.to_node
        )
        order = np.argsort(keys)
        self._edge_lookup_keys = keys[order]
        self._edge_lookup_values = order.astype(np.int32)

        # Vehicle indices ordered by start time, so that the vehicles starting
        # at a given step form a contiguous slice found by bisection.
        self._start_order = np.argsort(
            self._vehicles.start_time, kind='stable'
        ).astype(np.int32)
        self._sorted_start_times = self._vehicles.start_time[self._start_order]

        # Sorted indices of the vehicles currently in the network (AT_NODE or
        # IN_EDGE): per-step work scales with this set, not with the total
        # number of vehicles.
        self._active = np.empty(0, dtype=np.int32)

        self._next_leg: np.ndarray | None = None
        # Row index into _next_leg for each destination node (-1 when the node
        # is not a destination of any vehicle).
        self._dest_row: np.ndarray | None = None
        # Unique destination nodes: the Dijkstra sources (from/to inverted).
        self._route_targets: np.ndarray | None = None
        # Last step for which _next_leg is guaranteed fresh. Bounded only by
        # horizon-limited refreshes; see step(horizon=...).
        self._route_valid_until = Simulator.END_OF_TIME
        self._ready = False
        self._now = 0

    def _shortest_paths_to(self, targets: np.ndarray, travel_time) -> np.ndarray:
        """Compute next-hop predecessor rows towards each target node.

        Note that from/to are inverted so that Dijkstra, run from each target,
        returns the *successor* node on the path from any node to that target.

        When ``refresh_workers`` is configured, the targets are split over a
        persistent pool of worker processes; each target's computation is
        independent, so the result is identical to the in-process one.

        Args:
            targets: Destination node indices (the Dijkstra sources).
            travel_time: Per-edge travel time to use as the edge weight.

        Returns:
            Predecessor matrix of shape ``(len(targets), nr_nodes)``.
        """
        workers = self._refresh_workers
        if workers is not None and workers > 1 and len(targets) > 1:
            if self._refresh_pool is None:
                self._refresh_pool = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_refresh_pool_initializer,
                    initargs=(self._edges.to_node, self._edges.from_node, self._n_nodes),
                )
            travel_time = np.asarray(travel_time)
            chunks = [c for c in np.array_split(targets, workers) if len(c) > 0]
            predecessors = self._refresh_pool.map(
                _refresh_pool_worker, [(travel_time, c) for c in chunks]
            )
            return np.vstack(list(predecessors))

        graph = csr_array(
            (travel_time, (self._edges.to_node, self._edges.from_node)),
            shape=(self._n_nodes, self._n_nodes),
        )
        _, predecessors = shortest_path_search(
            graph, return_predecessors=True, indices=targets
        )
        return predecessors

    def close(self) -> None:
        """Release resources held by the simulator.

        Shuts down the worker-process pool used for parallel shortest-path
        recomputation, if one was started. The simulator remains usable; a
        new pool is started on demand.
        """
        if self._refresh_pool is not None:
            self._refresh_pool.shutdown()
            self._refresh_pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _edges_between(self, from_nodes: np.ndarray, to_nodes: np.ndarray) -> np.ndarray:
        """Return the edge index for each (from, to) node pair.

        All requested pairs must correspond to existing edges (as is the case
        for consecutive nodes on a shortest path).
        """
        keys = from_nodes.astype(np.int64) * self._n_nodes + to_nodes
        pos = np.searchsorted(self._edge_lookup_keys, keys)
        assert (pos < len(self._edge_lookup_keys)).all()
        assert (self._edge_lookup_keys[pos] == keys).all()
        return self._edge_lookup_values[pos]

    @classmethod
    def build(
        cls,
        edges,
        vehicles,
        *,
        fix_unreachable: bool = False,
        random: bool = False,
        seed: int | np.random.Generator | None = None,
        refresh_interval: int | None = None,
        refresh_workers: int | None = None,
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
            seed: Seed (or ready-made :class:`numpy.random.Generator`) for the
                shuffling performed when ``random=True``. Passing an integer
                makes random runs reproducible. Defaults to ``None``
                (fresh entropy on every run).
            refresh_interval: If set, :meth:`step` triggers the shortest-path
                recomputation automatically every ``refresh_interval`` steps
                (with the matching horizon), so callers no longer need to pass
                ``update_next_leg``/``horizon``. Defaults to ``None`` (caller
                -driven refreshes only).
            refresh_workers: If set to 2 or more, shortest-path recomputations
                are distributed over a persistent pool of that many worker
                processes. Results are identical to the single-process
                computation. Defaults to ``None`` (compute in-process).

        Returns:
            A two-tuple ``(simulator, fixed)`` where *simulator* is ready to
            call :meth:`step` on, and *fixed* is the list of vehicle indices
            whose destination was reset to their origin (empty when nothing
            needed fixing).

        Raises:
            ValueError: If ``fix_unreachable=False`` and one or more vehicles
                have unreachable destinations.
        """
        simulator = cls(
            edges, vehicles, random=random, seed=seed,
            refresh_interval=refresh_interval, refresh_workers=refresh_workers,
        )

        # Compute the routing with free-flow speeds. Dijkstra runs only from
        # the nodes that are actual vehicle destinations, so _next_leg has one
        # row per unique destination (indexed via _dest_row) instead of one
        # row per node. All destination rows are needed here (not only those
        # within some departure horizon) because the reachability check below
        # inspects every vehicle regardless of its start time.
        simulator._route_targets = np.unique(simulator._vehicles.destination)
        simulator._dest_row = np.full(simulator._n_nodes, -1, dtype=np.int32)
        simulator._dest_row[simulator._route_targets] = np.arange(
            len(simulator._route_targets), dtype=np.int32
        )
        simulator._next_leg = simulator._shortest_paths_to(
            simulator._route_targets, edges['length'] / edges['speed']
        )

        # Check for vehicles with unreachable destinations
        unreachable_mask = (
            (simulator._vehicles.destination != simulator._vehicles.origin) &
            (simulator._next_leg[
                simulator._dest_row[simulator._vehicles.destination],
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
            # The new destination may have no _dest_row entry, but these
            # vehicles arrive at start (node == destination) without ever
            # querying the routing.
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

    def step(self, update_next_leg: bool = False, horizon: int | None = None) -> None:
        """Advance the simulation by one time step.

        The method updates vehicle states in the following order:

        1. **Starting** — vehicles whose departure time equals the current step
           transition from ``WAITING`` to ``AT_NODE``.
        2. **Out-of-edge** — vehicles that have travelled the full length of
           their edge transition from ``IN_EDGE`` to ``AT_NODE``.
        3. **Arrived** — vehicles at their destination node transition to
           ``ARRIVED``.
        4. **Entering edge** — vehicles at a node attempt to enter the next
           edge on their shortest path. Admission is decided against the lane
           state at the beginning of this phase: each lane with enough room
           admits one vehicle, and lanes vacated during this same step become
           available only in the next step.
        5. **Progress** — vehicles in edges advance by their edge speed, up
           to the front vehicle's position minus ``DELTA``.

        When the simulator was built with ``refresh_interval``, the
        shortest-path recomputation is triggered automatically every
        ``refresh_interval`` steps and these arguments are not needed.

        Args:
            update_next_leg: If ``True``, recompute shortest paths using
                instantaneous measured speeds after updating vehicle positions.
                This is expensive and is typically done only every few hundred
                steps. Defaults to ``False``.
            horizon: Upper bound, in steps, on the interval until the *next*
                shortest-path recomputation. When given, only the routing rows
                that can be read before then are recomputed: those of
                destinations of vehicles currently in the network or departing
                within ``horizon`` steps. This is much cheaper and produces
                identical results, provided the caller does recompute at least
                every ``horizon`` steps. Defaults to ``None`` (recompute all
                rows). Only meaningful together with ``update_next_leg=True``.

        Raises:
            RuntimeError: If the simulator was not built with :meth:`build`,
                or if routing data is read after a promised ``horizon`` has
                elapsed without the corresponding recomputation.
        """
        if not self._ready:
            raise RuntimeError(
                "Simulator.build() must be called before step(). "
                "Use Simulator.build(edges, vehicles) to create a ready simulator."
            )
        assert self._next_leg is not None
        next_leg: np.ndarray = self._next_leg

        # Automatic refresh: same cadence and semantics as a caller passing
        # update_next_leg=True with horizon=refresh_interval every
        # refresh_interval steps. An explicit update_next_leg call keeps its
        # own horizon.
        if not update_next_leg and self._refresh_interval is not None:
            if (self._now + 1) % self._refresh_interval == 0:
                update_next_leg = True
                horizon = self._refresh_interval

        waiting_status = Simulator.VehicleStatus.WAITING.value
        at_node_status = Simulator.VehicleStatus.AT_NODE.value
        in_edge_status = Simulator.VehicleStatus.IN_EDGE.value
        arrived_status = self.VehicleStatus.ARRIVED.value

        self._vehicles.started_now[:] = False
        self._vehicles.arrived_now[:] = False
        self._vehicles.entered_edge_now[:] = False

        def do_starting() -> None:
            # Find vehicles that are ready to start: a contiguous slice of the
            # start-time-ordered vehicle indices.
            # Search with matching dtype: passing Python ints would promote
            # (and copy) the whole sorted array to int64 on every step.
            lo, hi = np.searchsorted(
                self._sorted_start_times,
                np.array([self._now, self._now + 1], dtype=self._sorted_start_times.dtype),
            )
            starting = self._start_order[lo:hi]
            starting = starting[self._vehicles.status[starting] == waiting_status]
            if len(starting) > 0:
                self._vehicles.status[starting] = at_node_status
                self._vehicles.started_now[starting] = True
                # Merge into the sorted active-vehicle set
                starting = np.sort(starting)
                self._active = np.insert(
                    self._active, np.searchsorted(self._active, starting), starting
                )

        do_starting()

        def do_out_of_edge() -> None:
            # Find vehicles that have reached the end of their edge
            active = self._active
            in_edge_v = active[self._vehicles.status[active] == in_edge_status]
            vehicles = in_edge_v[
                self._vehicles.edge_distance[in_edge_v]
                == self._edges.length[self._vehicles.edge[in_edge_v]]
            ]
            self._vehicles.status[vehicles] = at_node_status

        do_out_of_edge()

        def do_arrived() -> None:
            # Find vehicles that reached their final destination
            active = self._active
            at_node_v = active[self._vehicles.status[active] == at_node_status]
            arrived = at_node_v[
                self._vehicles.node[at_node_v] == self._vehicles.destination[at_node_v]
            ]
            if len(arrived) > 0:
                self._vehicles.status[arrived] = arrived_status
                self._vehicles.arrival_time[arrived] = self._now
                self._vehicles.arrived_now[arrived] = True

                # Restrict to vehicles that did not start right away (origin == destination)
                arrived_not_started = arrived[self._vehicles.edge[arrived] >= 0]
                # Restrict to vehicles that are last in their lane
                in_out_vehicles = arrived_not_started[
                    self._edges.last_vehicle[
                        self._vehicles.edge[arrived_not_started],
                        self._vehicles.lane[arrived_not_started],
                    ] == arrived_not_started
                ]
                # Mark those lanes as empty
                self._edges.last_vehicle[
                    self._vehicles.edge[in_out_vehicles],
                    self._vehicles.lane[in_out_vehicles],
                ] = -1
                # Reset edge, lane and front pointer for arrived vehicles
                self._vehicles.edge[arrived_not_started] = -1
                self._vehicles.lane[arrived_not_started] = -1
                self._vehicles.lane_front_vehicle[arrived_not_started] = -1
                # Arrived vehicles leave the active set
                self._active = active[self._vehicles.status[active] != arrived_status]

        do_arrived()

        def do_entering_edge() -> None:
            # Find vehicles that should enter an edge (_active is sorted, so
            # candidates are in ascending vehicle-index order).
            # Note that from/to are inverted in _next_leg.
            candidates = self._active[
                self._vehicles.status[self._active] == at_node_status
            ]
            if len(candidates) == 0:
                return
            if self._now > self._route_valid_until:
                raise RuntimeError(
                    "Routing data may be stale: the last shortest-path "
                    f"recomputation promised a refresh within its horizon "
                    f"(by step {self._route_valid_until}), but none happened "
                    f"by step {self._now}. Call step(update_next_leg=True) "
                    "at least every `horizon` steps, or pass horizon=None."
                )
            dest_rows = self._dest_row[self._vehicles.destination[candidates]]
            assert (dest_rows >= 0).all()
            next_nodes = next_leg[
                dest_rows,
                self._vehicles.node[candidates],
            ]

            assert not (next_nodes == -9999).any()

            next_edges = self._edges_between(
                self._vehicles.node[candidates], next_nodes
            )

            # Priority among vehicles competing for the same edge: vehicle
            # index order, or a random order for random simulations.
            if self._random:
                priority = self._rng.permutation(len(candidates))
                candidates = candidates[priority]
                next_nodes = next_nodes[priority]
                next_edges = next_edges[priority]

            # Group the candidates by target edge, keeping the priority order
            # within each group.
            grouping = np.argsort(next_edges, kind='stable')
            candidates = candidates[grouping]
            next_nodes = next_nodes[grouping]
            next_edges = next_edges[grouping]
            unique_edges, group_start, group_size = np.unique(
                next_edges, return_index=True, return_counts=True
            )

            # Lane gaps as of the beginning of this phase: lanes vacated by
            # vehicles moving on during this same step only become available
            # in the next step.
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

            # Each lane with a gap of at least DELTA admits exactly one
            # vehicle (it enters at distance 0, leaving no room for another).
            # Per edge, the first free_lanes[group] candidates in priority
            # order are admitted, filling the qualifying lanes in
            # decreasing-gap order (ties broken by lane index).
            free_lanes = (lane_distances >= Simulator.DELTA).sum(axis=1)
            group_id = np.repeat(np.arange(len(unique_edges)), group_size)
            rank = np.arange(len(candidates)) - np.repeat(group_start, group_size)
            admitted = rank < free_lanes[group_id]

            if not admitted.any():
                return

            vehicles = candidates[admitted]
            edges = next_edges[admitted]
            lane_order = np.argsort(-lane_distances, axis=1, kind='stable')
            lanes = lane_order[group_id[admitted], rank[admitted]]
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
            self._vehicles.node[vehicles] = next_nodes[admitted]
            self._vehicles.edge_distance[vehicles] = 0.0
            self._vehicles.lane_front_vehicle[vehicles] = self._edges.last_vehicle[edges, lanes]
            self._edges.last_vehicle[edges, lanes] = vehicles
            np.add.at(self._edges.nr_vehicles, edges, 1)

        do_entering_edge()

        def do_progress_vehicles() -> None:
            active = self._active
            # Clear stale front-vehicle references (front vehicle has changed edge)
            front_ref = self._vehicles.lane_front_vehicle[active]
            new_edge_front = active[
                (front_ref != -1) &
                (self._vehicles.edge[active] != self._vehicles.edge[front_ref])
            ]
            self._vehicles.lane_front_vehicle[new_edge_front] = -1

            in_edge_v = active[self._vehicles.status[active] == in_edge_status]
            front_of_lane = self._vehicles.lane_front_vehicle[in_edge_v] == -1
            in_edge_front = in_edge_v[front_of_lane]
            in_edge_follow = in_edge_v[~front_of_lane]

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
                    travel_time = self._edges.length / instant_speed
                    if horizon is None:
                        self._next_leg = self._shortest_paths_to(
                            self._route_targets, travel_time
                        )
                        self._route_valid_until = Simulator.END_OF_TIME
                        return
                    # Restrict the recomputation to the routing rows that can
                    # be read before the next recomputation: destinations of
                    # vehicles in the network now or departing within horizon
                    # steps. The remaining rows are recomputed again before
                    # any vehicle reads them.
                    lo, hi = np.searchsorted(
                        self._sorted_start_times,
                        np.array(
                            [self._now, self._now + horizon],
                            dtype=self._sorted_start_times.dtype,
                        ),
                        side='right',
                    )
                    targets = np.unique(np.concatenate([
                        self._vehicles.destination[self._active],
                        self._vehicles.destination[self._start_order[lo:hi]],
                    ]))
                    # Drop destinations without a routing row (vehicles fixed
                    # by fix_unreachable arrive at start and never route)
                    targets = targets[self._dest_row[targets] >= 0]
                    self._route_valid_until = self._now + horizon
                    if len(targets) == 0:
                        return
                    self._next_leg[self._dest_row[targets]] = (
                        self._shortest_paths_to(targets, travel_time)
                    )

                do_update_next_leg()

            self._vehicles.edge_distance[in_edge_follow] = new_distance

            # Leading vehicles advance up to the edge length
            self._vehicles.edge_distance[in_edge_front] = np.minimum(
                self._vehicles.edge_distance[in_edge_front]
                + self._edges.speed[self._vehicles.edge[in_edge_front]],
                self._edges.length[self._vehicles.edge[in_edge_front]],
            )

        do_progress_vehicles()

        assert not (self._vehicles.edge_distance[self._active] < 0.0).any()

        self._now += 1
