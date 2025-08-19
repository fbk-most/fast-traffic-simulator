from dataclasses import dataclass, field
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

@dataclass
class CarFollowingQueue:
    max_length: np.uint16
    distances: np.ndarray = field(init=False)
    vehicles: np.ndarray = field(init=False)
    start: np.uint16 = 0
    length: np.uint16 = 0

    def __post_init__(self):
        self.distances = np.full(self.max_length, -1, dtype=np.float32)
        self.vehicles = np.full(self.max_length, -1, dtype=np.int32)

    def is_empty(self) -> bool:
        return self.length == 0

    def is_full(self) -> bool:
        return self.length == self.max_length

    def add(self, vehicle: np.int32):
        if self.start + self.length == self.max_length:
            if self.start == 0:
                raise RuntimeError('Queue is full')
            self.distances[:self.length] = self.distances[self.start:]
            self.vehicles[:self.length] = self.vehicles[self.start:]
            self.start = 0
        self.distances[self.start + self.length] = 0.0
        self.vehicles[self.start + self.length] = vehicle
        self.length += 1

    def get_tail_distance(self) -> np.float32:
        if self.length == 0:
            raise RuntimeError('Queue is empty')
        return self.distances[self.start + self.length - 1]

    def get_head_distance(self) -> np.float32:
        if self.length == 0:
            raise RuntimeError('Queue is empty')
        return self.distances[self.start]

    def remove(self) -> np.int32:
        if self.length == 0:
            raise RuntimeError('Queue is empty')
        vehicle = self.vehicles[self.start]
        self.length -= 1
        self.start = 0 if self.length == 0 else self.start+1
        return vehicle

    def step(self, speed: np.float32, delta: np.float32, max_value: np.float32):
        if self.length == 0:
            return
        if self.length > 1:
            self.distances[self.start + 1:self.start + self.length] = np.minimum(
                self.distances[self.start + 1:self.start + self.length] + speed,
                self.distances[self.start:self.start + self.length - 1] - delta
            )
        self.distances[self.start] = np.minimum(
            self.distances[self.start] + speed,
            max_value
        )

class Simulator:
    MAX_DENSITY: np.float32 = 0.2
    DELTA: np.float32 = 1.0 / MAX_DENSITY
    END_OF_TIME = 999_999

    @dataclass
    class VehiclesRecord:
        origin: np.ndarray
        destination: np.ndarray
        status: np.ndarray
        node: np.ndarray
        edge: np.ndarray
        start_time: np.ndarray
        arrival_time: np.ndarray

    class VehicleStatus(Enum):
        WAITING = auto()
        AT_NODE = auto()
        IN_EDGE = auto()
        ARRIVED = auto()

    @dataclass
    class EdgesRecord:
        from_node: list[np.int32]
        to_node: list[np.int32]
        length: list[np.int32]
        speed: list[np.float32]
        lanes: list[np.int32]
        queue: list[CarFollowingQueue]


    def __init__(self, edges, vehicles):
        self._nr_vehicles = vehicles.shape[0]
        self._vehicles = Simulator.VehiclesRecord(
            status = np.full(self._nr_vehicles, self.VehicleStatus.WAITING.value, dtype=np.int32),
            node = vehicles['origin'].values.astype(np.int32),
            edge = np.full(self._nr_vehicles, -1, dtype=np.int32),
            origin = vehicles['origin'].values.astype(np.int32),
            destination = vehicles['destination'].values.astype(np.int32),
            start_time = vehicles['start'].values.astype(np.int32),
            arrival_time = np.full(self._nr_vehicles, Simulator.END_OF_TIME, dtype=np.int32)
        )

        self._nr_edges = edges.shape[0]
        capacity = np.maximum(np.ceil(edges['length'] * edges['lanes'] * Simulator.MAX_DENSITY).values, 1).astype(np.int32)
        self._edges = Simulator.EdgesRecord(
            from_node=edges['from'].values.astype(np.int32),
            to_node=edges['to'].values.astype(np.int32),
            length=edges['length'].values.astype(np.int32),
            speed=edges['speed'].values.astype(np.float32),
            lanes=edges['lanes'].values.astype(np.int32),
            queue=np.array([CarFollowingQueue(c) for c in capacity]),
        )

        self._nodes_to_edge_map = coo_array((range(self._nr_edges),(edges['from'], edges['to']))).toarray()

        self._now = 0

    def step(self, routing):
        # Get the numeric values of the statuses
        waiting_status = self.VehicleStatus.WAITING.value
        at_node_status = self.VehicleStatus.AT_NODE.value
        in_edge_status = self.VehicleStatus.IN_EDGE.value
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
            out_of_edge = [ self._edges.queue[e].remove()
                            for e in range(self._nr_edges)
                            if (self._edges.queue[e].length and
                                self._edges.queue[e].get_head_distance() == self._edges.length[e]) ]
            # Update their status to AT_NODE
            self._vehicles.status[out_of_edge] = at_node_status
        do_out_of_edge()

        def do_arrived():
            # Find vehicles that reached their final destination
            arrived = ((self._vehicles.status == at_node_status) & (self._vehicles.node == self._vehicles.destination))
            # Update their status to ARRIVED
            if np.any(arrived):
                self._vehicles.status[arrived] = arrived_status
                self._vehicles.arrival_time[arrived] = self._now
        do_arrived()

        def do_entering_edge():
            # Find vehicles that should enter an edge
            entering_edge = (self._vehicles.status == at_node_status)
            # Update their status to IN_EDGE (and define the next node)
            if np.any(entering_edge):
                next_node, dist = routing.next_leg(self._vehicles.node[entering_edge],
                                                   self._vehicles.destination[entering_edge])
                edge = self._nodes_to_edge_map[self._vehicles.node[entering_edge], next_node]
                unique_edges, index, counts = np.unique(edge, return_index=True, return_counts=True)

                for e,i in zip(unique_edges, index):
                    if (self._edges.queue[e].length == 0 or
                            self._edges.queue[e].get_tail_distance() >= Simulator.DELTA):
                        # selected = entering_edge.nonzero()[0][index][available_edges]
                        v = entering_edge.nonzero()[0][i]
                        self._edges.queue[e].add(v)
                        self._vehicles.status[v] = in_edge_status
                        self._vehicles.edge[v] = e
                        self._vehicles.node[v] = next_node[i] ### TODO: -1?
        do_entering_edge()

        def do_edges():
            for e in range(self._nr_edges):
                if self._edges.queue[e].length:
                    self._edges.queue[e].step(self._edges.speed[e], Simulator.DELTA, self._edges.length[e])
            # xxx(self._edges.queue, self._edges.speed, Simulator.DELTA, self._edges.length)
        do_edges()

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
    vehicles = vehicles.head(10000)
    ### vehicles['start'] = 0

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

    travel_times = simulator._vehicles.arrival_time - converted_vehicles['start'].values
    print("\n--- Travel Statistics ---")
    print(f"Average travel time: {travel_times.mean():.2f} seconds")
    print(f"Minimum travel time: {travel_times.min():.2f} seconds")
    print(f"Maximum travel time: {travel_times.max():.2f} seconds")

    if not checks_ok:
        print("\n#### Some checks failed ####")