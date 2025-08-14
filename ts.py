import numpy as np
import pandas as pd
from scipy.sparse import coo_array
from scipy.sparse.csgraph import johnson
from enum import Enum, auto

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
    class VehicleStatus(Enum):
        WAITING = auto()
        AT_NODE = auto()
        IN_EDGE = auto()
        ARRIVED = auto()

    END_OF_TIME = 999_999

    def __init__(self, vehicles):
        nr_vehicles = vehicles.shape[0]

        # We could use self.VehicleStatus as type for status below, but performance would be much worse
        # Moreover, if we ensure alignment, performance is higher
        self._vstatus = np.zeros(nr_vehicles,
                                 dtype=[('status', np.int32),
                                        ('node', np.int32),
                                        ('prev_node', np.int32),
                                        ('destination', np.int32),
                                        ('timer', np.int32)])

        self._vstatus['status'] = self.VehicleStatus.WAITING.value
        self._vstatus['node'] = vehicles['origin'].values
        self._vstatus['prev_node'] = -1
        self._vstatus['destination'] = vehicles['destination'].values
        self._vstatus['timer'] = vehicles['start'].values

        self._timer = 0

    def step(self, routing):
        # Get the numeric values of the statuses
        waiting_status = self.VehicleStatus.WAITING.value
        at_node_status = self.VehicleStatus.AT_NODE.value
        in_edge_status = self.VehicleStatus.IN_EDGE.value
        arrived_status = self.VehicleStatus.ARRIVED.value

        right_time = (self._vstatus['timer'] == self._timer)

        def do_starting():
            # Find vehicles to be updated
            starting = right_time & (self._vstatus['status'] == waiting_status)
            # Update their status to AT_NODE
            if np.any(starting):
                self._vstatus['status'][starting] = at_node_status
        do_starting()

        def do_out_of_edge():
            # Find vehicles that arrive at a node
            out_of_edge = right_time & (self._vstatus['status'] == in_edge_status)
            # Update their status to AT_NODE
            if np.any(out_of_edge):
                self._vstatus['status'][out_of_edge] = at_node_status
        do_out_of_edge()

        def do_arrived():
            # Find vehicles that reached their final destination
            arrived = (right_time & (self._vstatus['status'] == at_node_status) &
                       (self._vstatus['node'] == self._vstatus['destination']))
            # Update their status to ARRIVED
            if np.any(arrived):
                self._vstatus['status'][arrived] = arrived_status
                self._vstatus['timer'][arrived] = Simulator.END_OF_TIME
        do_arrived()

        def do_entering_edge():
            # Find vehicles that should enter an edge
            entering_edge = right_time & (self._vstatus['status'] == at_node_status)
            # Update their status to IN_EDGE (and define edge and next node)
            if np.any(entering_edge):
                next_node, dist = routing.next_leg(self._vstatus['node'][entering_edge],
                                                   self._vstatus['destination'][entering_edge])
                self._vstatus['status'][entering_edge] = in_edge_status
                self._vstatus['prev_node'][entering_edge] = self._vstatus['node'][entering_edge]
                self._vstatus['node'][entering_edge] = next_node
                self._vstatus['timer'][entering_edge] = self._timer + np.ceil(dist)
        do_entering_edge()

        # Increment the timer
        self._timer += 1


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
    simulator = Simulator(vehicles=converted_vehicles)

    # Main timing
    start_sim = time.time()
    h = 0
    while True:
        for _ in range(60*60):
            simulator.step(routing)
        h += 1
        nr_waiting = (simulator._vstatus['status'] == Simulator.VehicleStatus.WAITING.value).sum()
        nr_at_node = (simulator._vstatus['status'] == Simulator.VehicleStatus.AT_NODE.value).sum()
        nr_in_edge = (simulator._vstatus['status'] == Simulator.VehicleStatus.IN_EDGE.value).sum()
        nr_arrived = (simulator._vstatus['status'] == Simulator.VehicleStatus.ARRIVED.value).sum()
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