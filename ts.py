from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import johnson


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
        queue_distances: list[np.float32]
        queue_vehicles: list[np.int32]
        queue_start: list[np.int32]  # First cell of the queue (for a given edge)
        queue_end: list[np.int32]    # Last cell of the queue (or, better, next cell after the last)
        queue_front: list[np.int32]  # Front of the queue (first vehicle)
        queue_back: list[np.int32]   # Back of the queue (last vehicle)
        queue_speed: list[np.float32] = field(init=False)

        def __post_init__(self):
            self.queue_speed = np.full_like(self.queue_distances, 0.0, dtype=np.float32)
            for i,s in enumerate(self.speed):
                self.queue_speed[self.queue_start[i]:self.queue_end[i]] = s

    def __init__(self, edges, vehicles, *, random=False):
        self._random = random
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
        total_capacity = capacity.sum()
        queue_end = capacity.cumsum()
        queue_start = np.roll(queue_end, 1); queue_start[0] = 0
        self._edges = Simulator.EdgesRecord(
            from_node=edges['from'].values.astype(np.int32),
            to_node=edges['to'].values.astype(np.int32),
            length=edges['length'].values.astype(np.int32),
            speed=edges['speed'].values.astype(np.float32),
            lanes=edges['lanes'].values.astype(np.int32),
            queue_distances=np.full((total_capacity+1,), 0.0, dtype=np.float32),
            queue_vehicles=np.full((total_capacity+1,), -1, dtype=np.int32),
            queue_start=queue_start,
            queue_end=queue_end,
            queue_front=queue_start.copy(),
            queue_back=queue_start.copy(),
        )

        self._nodes_to_edge_map = coo_array((range(self._nr_edges),(edges['from'], edges['to']))).toarray()

        # Compute the routing
        # Note that, in the following graph, from and to are inverted,
        # so that the Johnson's algorithm returns the successor node
        graph = coo_array((edges['length'] / edges['speed'], (edges['to'], edges['from']))).toarray()
        # Run the Johnson's algorith
        _, self._next_leg = johnson(graph, return_predecessors=True)

        self._now = 0

    def step(self):
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
            out_of_edge = ((self._edges.queue_front != self._edges.queue_back) &
                           (self._edges.queue_distances[self._edges.queue_front] == self._edges.length))
            vehicles = self._edges.queue_vehicles[self._edges.queue_front[out_of_edge]]
            self._edges.queue_front[out_of_edge] += 1
            # Reset empty queues (commented, as performance improvement is not clear)
            ### empty = (out_of_edge & (self._edges.queue_front == self._edges.queue_back))
            ### self._edges.queue_front[empty] = self._edges.queue_start[empty]
            ### self._edges.queue_back[empty] = self._edges.queue_start[empty]
            # Update their status to AT_NODE
            self._vehicles.status[vehicles] = at_node_status
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
                # Note that from and to are inverted
                next_node = self._next_leg[self._vehicles.destination[entering_edge],
                                           self._vehicles.node[entering_edge]]
                edge = self._nodes_to_edge_map[self._vehicles.node[entering_edge], next_node]

                if self._random:
                    order = np.arange(len(edge))
                    np.random.shuffle(order)
                    unique_edges, index = np.unique(edge[order], return_index=True)
                    index = order[index]
                else:
                    unique_edges, index = np.unique(edge, return_index=True)

                free_edges = ((self._edges.queue_front[unique_edges] == self._edges.queue_back[unique_edges]) |
                              (self._edges.queue_distances[self._edges.queue_back[unique_edges]-1] >= Simulator.DELTA))
                to_shift = (free_edges & (self._edges.queue_back[unique_edges] == self._edges.queue_end[unique_edges]))
                if np.any(to_shift):
                    full = (to_shift & (self._edges.queue_front[unique_edges] == self._edges.queue_start[unique_edges]))
                    if np.any(full):
                        raise RuntimeError('Queue is full')
                    for e in unique_edges[to_shift]:
                        self._edges.queue_back[e] -= self._edges.queue_front[e] - self._edges.queue_start[e]
                        self._edges.queue_distances[self._edges.queue_start[e]:self._edges.queue_back[e]] = \
                            self._edges.queue_distances[self._edges.queue_front[e]:self._edges.queue_end[e]]
                        self._edges.queue_vehicles[self._edges.queue_start[e]:self._edges.queue_back[e]] = \
                            self._edges.queue_vehicles[self._edges.queue_front[e]:self._edges.queue_end[e]]
                        self._edges.queue_front[e] = self._edges.queue_start[e]
                free_unique_edges = unique_edges[free_edges]
                vehicles = entering_edge.nonzero()[0][index[free_edges]]

                self._edges.queue_distances[self._edges.queue_back[free_unique_edges]] = 0.0
                self._edges.queue_vehicles[self._edges.queue_back[free_unique_edges]] = vehicles
                self._edges.queue_back[free_unique_edges] += 1
                self._vehicles.status[vehicles] = in_edge_status
                self._vehicles.edge[vehicles] = free_unique_edges
                self._vehicles.node[vehicles] = next_node[index[free_edges]]
        do_entering_edge()

        def do_edges():
            fronts = np.minimum(
                self._edges.queue_distances[self._edges.queue_front] + self._edges.speed,
                self._edges.length
            )
            self._edges.queue_distances[1:] = np.minimum(
                self._edges.queue_distances[1:] + self._edges.queue_speed[1:],
                self._edges.queue_distances[:-1] - Simulator.DELTA
            )
            self._edges.queue_distances[self._edges.queue_front] = fronts
        do_edges()

        # Increment the timer
        self._now += 1


if __name__ == '__main__':
    # Switch between randomised and deterministc behavior of he simulator
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
        for _ in range(60*60):
            simulator.step()
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
