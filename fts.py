from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import dijkstra as shortest_path_search # dijkstra / johnson


class Simulator:
    MAX_DENSITY: np.float32 = 0.2
    DELTA: np.float32 = 1.0 / MAX_DENSITY
    END_OF_TIME = 999_999


    class VehicleStatus(Enum):
        WAITING = auto()
        AT_NODE = auto()
        IN_EDGE = auto()
        ARRIVED = auto()


    @dataclass
    class VehiclesRecord:
        origin: list[np.int32]
        destination: list[np.int32]
        start_time: list[np.int32]
        status: list[np.int32] = field(init=False)
        node: list[np.int32] = field(init=False)
        edge: list[np.int32] = field(init=False)
        lane: list[np.int32] = field(init=False)
        edge_distance: list[np.float32] = field(init=False)
        lane_front_vehicle: list[np.int32] = field(init=False)
        arrival_time: list[np.int32] = field(init=False)

        def __post_init__(self):
            nr_vehicles = self.origin.shape[0]
            self.status=np.full(nr_vehicles, Simulator.VehicleStatus.WAITING.value, dtype=np.int32)
            self.node=self.origin.copy()
            self.edge=np.full(nr_vehicles, -1, dtype=np.int32)
            self.lane=np.full(nr_vehicles, -1, dtype=np.int32)
            self.lane_front_vehicle=np.full(nr_vehicles, -1, dtype=np.int32)
            self.edge_distance=np.full(nr_vehicles, 0.0, dtype=np.float32)
            self.arrival_time=np.full(nr_vehicles, Simulator.END_OF_TIME, dtype=np.int32)


    @dataclass
    class EdgesRecord:
        from_node: list[np.int32]
        to_node: list[np.int32]
        length: list[np.float32]
        speed: list[np.float32]
        lanes: list[np.int32]
        last_vehicle: np.ndarray = field(init=False)

        def __post_init__(self):
            nr_edges = self.from_node.shape[0]
            self.last_vehicle = np.empty((nr_edges, self.lanes.max()), dtype=np.int32)
            for l in range(self.lanes.max()):
                self.last_vehicle[:, l] = np.where(self.lanes > l, -1, -2)


    def __init__(self, edges, vehicles, *, random=False):
        self._random = random
        self._nr_vehicles = vehicles.shape[0]
        self._vehicles = Simulator.VehiclesRecord(
            origin = vehicles['origin'].values.astype(np.int32),
            destination = vehicles['destination'].values.astype(np.int32),
            start_time = vehicles['start'].values.astype(np.int32),
        )

        self._nr_edges = edges.shape[0]
        self._edges = Simulator.EdgesRecord(
            from_node=edges['from'].values.astype(np.int32),
            to_node=edges['to'].values.astype(np.int32),
            length=edges['length'].values.astype(np.float32),
            speed=edges['speed'].values.astype(np.float32),
            lanes=edges['lanes'].values.astype(np.int32),
        )

        self._max_lanes = self._edges.lanes.max()

        self._nodes_to_edge_map = coo_array((range(self._nr_edges),(edges['from'], edges['to']))).toarray()

        # Compute the routing
        # Note that, in the following graph, from and to are inverted,
        # so that the Dijkstra algorithm returns the successor node
        graph = coo_array((edges['length'] / edges['speed'], (edges['to'], edges['from']))).toarray()
        # Run the Dijkstra algorith
        _, self._next_leg = shortest_path_search(graph, return_predecessors=True)

        self._now = 0

    def step(self, update_next_leg = False):
        # Get the numeric values of the statuses
        waiting_status = Simulator.VehicleStatus.WAITING.value
        at_node_status = Simulator.VehicleStatus.AT_NODE.value
        in_edge_status = Simulator.VehicleStatus.IN_EDGE.value
        arrived_status = self.VehicleStatus.ARRIVED.value

        def do_starting():
            # Find vehicles that are ready to start
            starting = (self._vehicles.start_time == self._now) & (self._vehicles.status == waiting_status)
            # Update their status to AT_NODE
            if np.any(starting):
                self._vehicles.status[starting] = at_node_status
        do_starting()

        def do_out_of_edge():
            # Find vehicles that arrive at a node
            vehicles = ((self._vehicles.status == in_edge_status) &
                        (self._vehicles.edge_distance == self._edges.length[self._vehicles.edge]))
            # Update their status to AT_NODE
            self._vehicles.status[vehicles] = at_node_status
        do_out_of_edge()

        def do_arrived():
            # Find vehicles that reached their final destination
            arrived = ((self._vehicles.status == at_node_status) & (self._vehicles.node == self._vehicles.destination))
            if np.any(arrived):
                # Update their status to ARRIVED and save arrival time
                self._vehicles.status[arrived] = arrived_status
                self._vehicles.arrival_time[arrived] = self._now
                # Restrict to vehicles that did not start right away (ie started and arrived at the same node)
                arrived_not_started = arrived & (self._vehicles.edge >= 0)
                # Restrict to vehicles that are last in their lane
                in_out_vehicles = (self._edges.last_vehicle[
                                       self._vehicles.edge[arrived_not_started],
                                       self._vehicles.lane[arrived_not_started]] == arrived_not_started.nonzero()[0])
                in_out_vehicles = arrived_not_started.nonzero()[0][in_out_vehicles]
                # There are now no vehicles in that lane
                self._edges.last_vehicle[self._vehicles.edge[in_out_vehicles], self._vehicles.lane[in_out_vehicles]] = -1
                # Reset edge and lane  TODO: use arrived instead of arrived_not_started?
                self._vehicles.edge[arrived_not_started] = -1
                self._vehicles.lane[arrived_not_started] = -1
        do_arrived()

        def do_entering_edge():
            while True:
                # Find vehicles that should enter an edge
                entering_edge = (self._vehicles.status == at_node_status)
                if not np.any(entering_edge):
                    break
                # Compute the next node and edge
                # Note that from and to are inverted
                next_nodes = self._next_leg[self._vehicles.destination[entering_edge],
                                           self._vehicles.node[entering_edge]]
                next_edges = self._nodes_to_edge_map[self._vehicles.node[entering_edge], next_nodes]

                assert not (next_nodes == -9999).any()

                # Get unique edges aimed by the vehicles, and the corresponding vehicles
                # Note: shuffling is performed in case of random simulation
                if self._random:
                    order = np.arange(len(next_edges))
                    np.random.shuffle(order)
                    unique_edges, unique_vehicles = np.unique(next_edges[order], return_index=True)
                    unique_vehicles = order[unique_vehicles]
                else:
                    unique_edges, unique_vehicles = np.unique(next_edges, return_index=True)

                # Restrict to free edges and corresponding vehicles
                lane_distances = np.select(
                    [self._edges.last_vehicle[unique_edges] == -1, self._edges.last_vehicle[unique_edges] < -1, True],
                    [999_999.9, 0.0, self._vehicles.edge_distance[self._edges.last_vehicle[unique_edges]]])
                free_edges = (lane_distances >= Simulator.DELTA).any(axis=1)
                # Alternative code (note: also lanes computation needs to be changed)
                # empty_lanes = (self._edges.last_vehicle[unique_edges] == -1)
                # clear_lanes = ((self._edges.last_vehicle[unique_edges] >= 0) &
                #                (self._vehicles.edge_distance[self._edges.last_vehicle[unique_edges]]
                #                 >= Simulator.DELTA))
                # free_edges = (empty_lanes | clear_lanes).any(axis=1)

                if not free_edges.any():
                    break

                edges = unique_edges[free_edges]
                lanes = np.argmax(lane_distances[free_edges], axis=1)
                # Alternative code (if free_edges is changed)
                # lanes = np.argmax(np.select(
                #     [self._edges.last_vehicle[edges] == -1, self._edges.last_vehicle[edges] < -1, True],
                #     [999_999.9, 0.0, self._vehicles.edge_distance[self._edges.last_vehicle[edges]]]), axis=1)
                vehicles = entering_edge.nonzero()[0][unique_vehicles[free_edges]]
                assert not (self._edges.last_vehicle[edges, lanes] < -1).any()

                # If vehicles are last in their lane, them the lane in now empty
                in_out_vehicles = ((self._vehicles.edge[vehicles] != -1) &
                                   (self._edges.last_vehicle[self._vehicles.edge[vehicles], self._vehicles.lane[vehicles]] == vehicles))
                self._edges.last_vehicle[self._vehicles.edge[vehicles[in_out_vehicles]], self._vehicles.lane[vehicles[in_out_vehicles]]] = -1
                # Update their status to IN_EDGE (and define the auxiliary fields for IN_EDGE vehicles)
                self._vehicles.status[vehicles] = in_edge_status
                self._vehicles.edge[vehicles] = edges
                self._vehicles.lane[vehicles] = lanes
                self._vehicles.node[vehicles] = next_nodes[unique_vehicles[free_edges]]
                self._vehicles.edge_distance[vehicles] = 0.0
                self._vehicles.lane_front_vehicle[vehicles] = self._edges.last_vehicle[edges, lanes]
                # Update new edge
                self._edges.last_vehicle[edges, lanes] = vehicles
        do_entering_edge()

        def do_progress_vehicles():
            new_edge_front = ((self._vehicles.lane_front_vehicle != -1) &
                              (self._vehicles.edge != self._vehicles.edge[self._vehicles.lane_front_vehicle]))
            self._vehicles.lane_front_vehicle[new_edge_front] = -1

            in_edge = (self._vehicles.status == in_edge_status)
            in_edge_front = (in_edge & (self._vehicles.lane_front_vehicle == -1))
            in_edge_follow = (in_edge & ~ in_edge_front)

            new_distance = np.minimum(
                self._vehicles.edge_distance[in_edge_follow] + self._edges.speed[self._vehicles.edge[in_edge_follow]],
                self._vehicles.edge_distance[self._vehicles.lane_front_vehicle[in_edge_follow]] - Simulator.DELTA,
            )

            if update_next_leg:
                def do_update_next_leg():
                    step = new_distance - self._vehicles.edge_distance[in_edge_follow]
                    step_edge = self._vehicles.edge[in_edge_follow]
                    instant_speed = self._edges.speed.copy()
                    instant_speed[step_edge] = 0.01
                    np.add.at(instant_speed, step_edge, step)
                    unique_edge, count_edge = np.unique(step_edge, return_counts=True)
                    np.divide.at(instant_speed, unique_edge, count_edge)
                    graph = coo_array((self._edges.length / instant_speed, (self._edges.to_node, self._edges.from_node))).toarray()
                    _, self._next_leg = shortest_path_search(graph, return_predecessors=True)
                do_update_next_leg()

            self._vehicles.edge_distance[in_edge_follow] = new_distance

            self._vehicles.edge_distance[in_edge_front] = np.minimum(
                self._vehicles.edge_distance[in_edge_front] + self._edges.speed[self._vehicles.edge[in_edge_front]],
                self._edges.length[self._vehicles.edge[in_edge_front]]
            )

        do_progress_vehicles()

        assert not (self._vehicles.edge_distance < 0.0).any()

        # Increment the timer
        self._now += 1