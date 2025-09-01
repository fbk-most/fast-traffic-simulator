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
                empty_lanes = (self._edges.last_vehicle[unique_edges] == -1)
                clear_lanes = ((self._edges.last_vehicle[unique_edges] >= 0) &
                               (self._vehicles.edge_distance[self._edges.last_vehicle[unique_edges]]
                                >= Simulator.DELTA))
                free_edges = (empty_lanes | clear_lanes).any(axis=1)
                if not free_edges.any():
                    break

                edges = unique_edges[free_edges]
                lanes = np.where(empty_lanes[free_edges].any(axis=1),
                                 np.argmax(empty_lanes[free_edges], axis=1),
                                 np.argmax(self._vehicles.edge_distance[self._edges.last_vehicle[edges]], axis=1))
                vehicles = entering_edge.nonzero()[0][unique_vehicles[free_edges]]

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

        # Increment the timer
        self._now += 1


if __name__ == '__main__':
    # Switch between randomised and deterministic behavior of the simulator
    RANDOM=False

    import time
    from legacy import *

    # Load data
    print("Loading data...")
    start_load = time.time()
    (nodes, edges, vehicles) = read_legacy('nodes__v1.csv', 'edges__v1.csv',
                                         'platoons_size=20_reduction=0.0__v1_seed_0.parquet')
    # TODO: Manage multi-edges
    edges.drop_duplicates(subset=['from', 'to'], inplace=True)
    load_time = time.time() - start_load
    print(f"Data loaded in {load_time:.2f} seconds")

    # For quick tests, the number of vehicles can be reduced and the start time can be lowered
    # vehicles = vehicles.head(100000)
    ### vehicles['start'] = 0

    # Convert data
    print("Converting data...")
    start_convert = time.time()
    nodes_map = map_nodes(nodes)
    converted_edges = convert_edges(edges, nodes_map)
    converted_vehicles = convert_vehicles(vehicles, nodes_map)
    convert_time = time.time() - start_convert
    print(f"Data converted in {convert_time:.2f} seconds")

    # Initialize the simulator
    print("Initializing simulator...")
    start_init = time.time()
    simulator = Simulator(vehicles=converted_vehicles, edges=converted_edges, random=RANDOM)
    init_time = time.time() - start_init
    print(f"Simulator initialized in {init_time:.2f} seconds")

    # Run simulation
    print("Starting simulation...")
    # Main timing
    start_sim = time.time()
    h = 0
    while True:
        for s in range(60*60):
            simulator.step(s % 300 == 299)
        h += 1
        nr_waiting = (simulator._vehicles.status == Simulator.VehicleStatus.WAITING.value).sum()
        nr_at_node = (simulator._vehicles.status == Simulator.VehicleStatus.AT_NODE.value).sum()
        nr_in_edge = (simulator._vehicles.status == Simulator.VehicleStatus.IN_EDGE.value).sum()
        nr_arrived = (simulator._vehicles.status == Simulator.VehicleStatus.ARRIVED.value).sum()
        print(f"... simulation time after {h} hours: {time.time()-start_sim:.2f} seconds")
        print(f"...... waiting: {nr_waiting} vehicles")
        print(f"...... at node: {nr_at_node} vehicles")
        print(f"...... in edge: {nr_in_edge} vehicles")
        print(f"...... arrived: {nr_arrived} vehicles")

        if nr_waiting + nr_at_node + nr_in_edge == 0:
            break

    sim_time = time.time() - start_sim
    total_steps = h*60*60
    steps_per_second = total_steps / sim_time

    print("\n--- Performance Summary ---")
    print(f"Total simulation time: {sim_time:.2f} seconds")
    print(f"Steps per second: {steps_per_second:.2f}")
    print(f"Time per step: {(sim_time/total_steps)*1000:.4f} ms")
    print(f"Total runtime: {load_time + convert_time + init_time + sim_time:.2f} seconds")

    travel_times = simulator._vehicles.arrival_time - converted_vehicles['start'].values
    print("\n--- Travel Statistics ---")
    print(f"Average travel time: {travel_times.mean():.2f} seconds")
    print(f"Minimum travel time: {travel_times.min():.2f} seconds")
    print(f"Maximum travel time: {travel_times.max():.2f} seconds")
