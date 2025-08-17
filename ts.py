from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import johnson

class Routing:
    def __init__(self, edges):
        self._edges = edges

        # Compute sparse graph
        # Note that from and to are inverted, so that the Johnson's algorithm returns the successor node
        self._graph = coo_array((edges['length'] / edges['speed'], (edges['to'], edges['from']))).toarray()

        # Run the Johnson's algorith
        _, self._next_node = johnson(self._graph, return_predecessors=True)


    def next_leg(self, node, destination):
        next_node = self._next_node[destination, node]
        dist = self._graph[next_node, node]

        return next_node, dist

class Simulator:
    MAX_DENSITY = 0.2
    END_OF_TIME = 999_999

    @dataclass
    class VehiclesRecord:
        origin: np.ndarray
        destination: np.ndarray
        status: np.ndarray
        node: np.ndarray
        prev_node: np.ndarray
        timer: np.ndarray

    class VehicleStatus(Enum):
        WAITING = auto()
        AT_NODE = auto()
        IN_EDGE = auto()
        ARRIVED = auto()

    @dataclass
    class EdgesRecord:
        from_node: np.ndarray
        to_node: np.ndarray
        length: np.ndarray
        speed: np.ndarray
        lanes: np.ndarray
        traversal_time: np.ndarray
        capacity: np.ndarray
        occupancy: np.ndarray


    def __init__(self, edges, vehicles):
        nr_vehicles = vehicles.shape[0]
        self._vehicles = Simulator.VehiclesRecord(
            status = np.full(nr_vehicles, self.VehicleStatus.WAITING.value, dtype=np.int32),
            node = vehicles['origin'].values.astype(np.int32),
            prev_node = np.full(nr_vehicles, -1, dtype=np.int32),
            origin = vehicles['origin'].values.astype(np.int32),
            destination = vehicles['destination'].values.astype(np.int32),
            timer = vehicles['start'].values.astype(np.int32),
        )

        nr_edges = edges.shape[0]
        self._edges = Simulator.EdgesRecord(
            from_node=edges['from'].values.astype(np.int32),
            to_node=edges['to'].values.astype(np.int32),
            length=edges['length'].values.astype(np.int32),
            speed=edges['speed'].values.astype(np.float32),
            lanes=edges['lanes'].values.astype(np.int32),
            traversal_time=(edges['length'] / edges['speed']).values.astype(np.int32),
            capacity=np.floor(edges['length'] * edges['lanes'] * Simulator.MAX_DENSITY).values.astype(np.int32),
            occupancy=np.zeros(nr_edges, dtype=np.int32),
        )

        self._nodes_to_edge_map = coo_array((range(nr_edges),(edges['from'], edges['to']))).toarray()

        self._now = 0

    def step(self, routing):
        # Get the numeric values of the statuses
        waiting_status = self.VehicleStatus.WAITING.value
        at_node_status = self.VehicleStatus.AT_NODE.value
        in_edge_status = self.VehicleStatus.IN_EDGE.value
        arrived_status = self.VehicleStatus.ARRIVED.value

        right_time = (self._vehicles.timer == self._now)  # All elements are compared to themselves, which is always True

        if not np.any(right_time):
            # Increment the timer and exit
            self._now += 1
            return

        def do_starting():
            # Find vehicles to be updated
            starting = right_time & (self._vehicles.status == waiting_status)
            # Update their status to AT_NODE
            if np.any(starting):
                self._vehicles.status[starting] = at_node_status
        do_starting()

        def do_out_of_edge():
            # Find vehicles that arrive at a node
            out_of_edge = right_time & (self._vehicles.status == in_edge_status)
            # Update their status to AT_NODE
            if np.any(out_of_edge):
                self._vehicles.status[out_of_edge] = at_node_status
                edge = self._nodes_to_edge_map[self._vehicles.prev_node[out_of_edge], self._vehicles.node[out_of_edge]]
                unique_edges, counts = np.unique(edge, return_counts=True)
                self._edges.occupancy[unique_edges] -= counts
        do_out_of_edge()

        def do_arrived():
            # Find vehicles that reached their final destination
            arrived = (right_time & (self._vehicles.status == at_node_status) &
                       (self._vehicles.node == self._vehicles.destination))
            # Update their status to ARRIVED
            if np.any(arrived):
                self._vehicles.status[arrived] = arrived_status
                self._vehicles.timer[arrived] = Simulator.END_OF_TIME
        do_arrived()

        def do_entering_edge():
            # Find vehicles that should enter an edge
            entering_edge = right_time & (self._vehicles.status == at_node_status)
            # Update their status to IN_EDGE (and define the next node)
            if np.any(entering_edge):
                next_node, dist = routing.next_leg(self._vehicles.node[entering_edge],
                                                   self._vehicles.destination[entering_edge])
                edge = self._nodes_to_edge_map[self._vehicles.node[entering_edge], next_node]
                unique_edges, counts = np.unique(edge, return_counts=True)
                self._edges.occupancy[unique_edges] += counts
                self._vehicles.status[entering_edge] = in_edge_status
                self._vehicles.prev_node[entering_edge] = self._vehicles.node[entering_edge]
                self._vehicles.node[entering_edge] = next_node
                self._vehicles.timer[entering_edge] = self._now + np.ceil(dist)
        do_entering_edge()

        # Increment the timer
        self._now += 1


if __name__ == '__main__':
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

    # For quick tests, the number of vehicles can be reduced
    # vehicles = vehicles.head(100000)

    # Convert data
    print("Converting data...")
    start_convert = time.time()
    nodes_map = map_nodes(nodes)
    converted_edges = convert_edges(edges, nodes_map)
    converted_vehicles = convert_vehicles(vehicles, nodes_map)
    convert_time = time.time() - start_convert
    print(f"Data converted in {convert_time:.2f} seconds")

    # Initialize routing
    print("Initializing routing...")
    start_routing = time.time()
    routing = Routing(edges=converted_edges)
    routing_time = time.time() - start_routing
    print(f"Routing initialized in {routing_time:.2f} seconds")
    
    # Run simulation
    print("Starting simulation...")
    simulator = Simulator(vehicles=converted_vehicles, edges=converted_edges)

    # Main timing
    start_sim = time.time()
    h = 0
    checks_ok = True
    while True:
        for _ in range(60*60):
            simulator.step(routing)
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

        # Checks
        nr_in_edge_from_occupancy = simulator._edges.occupancy.sum()
        if nr_in_edge_from_occupancy != nr_in_edge:
            print(f"###### Check failed: {nr_in_edge_from_occupancy} vehicles in edge from occupancy")
            checks_ok = False
        nr_edges_exceeding_capacity = (simulator._edges.occupancy > simulator._edges.capacity).sum()
        if nr_edges_exceeding_capacity != 0:
            print(f"###### Check failed: {nr_edges_exceeding_capacity} edge where occupancy exceeds capacity")
            checks_ok = False

        if nr_waiting + nr_at_node + nr_in_edge == 0:
            break

    sim_time = time.time() - start_sim
    total_steps = h*60*60
    steps_per_second = total_steps / sim_time

    print("\n--- Performance Summary ---")
    print(f"Total simulation time: {sim_time:.2f} seconds")
    print(f"Steps per second: {steps_per_second:.2f}")
    print(f"Time per step: {(sim_time/total_steps)*1000:.4f} ms")
    print(f"Total runtime: {load_time + convert_time + routing_time + sim_time:.2f} seconds")

    if not checks_ok:
        print("\n#### Some checks failed ####")