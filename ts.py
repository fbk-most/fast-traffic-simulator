import numpy as np
import pandas as pd
from scipy.sparse import coo_array
from scipy.sparse.csgraph import johnson
from enum import Enum, auto


def read_legacy(nodes_file, edges_file, platoons_file):
    nodes = pd.read_csv(nodes_file, names=['id', 'x', 'y', 'in-out', 'zone'], index_col='id')
    edges = pd.read_csv(edges_file, names=['id', 'from', 'to', 'length', 'speed', 'lanes', 'zone'], index_col='id')
    platoons = pd.read_parquet(platoons_file)[
        ['platoon_id', 'second', 'from_node_id', 'to_node_id', 'nr_vehicles']].set_index('platoon_id')
    nr_vehicles = platoons['nr_vehicles'].sum()
    platoons['last'] = platoons['nr_vehicles'].cumsum()
    platoons['first'] = platoons['last'] - platoons['nr_vehicles']
    v = np.zeros(nr_vehicles, dtype=[('from_node', np.int64), ('to_node', np.int64), ('start', np.int64)])
    for _, r in platoons.iterrows():
        v[int(r['first']):int(r['last'])] = (r['from_node_id'], r['to_node_id'], r['second'])
    vehicles = pd.DataFrame(v)
    vehicles.index.name = 'id'
    return nodes, edges, vehicles


class Routing:
    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges

        # Needed to associate nodes with an incremental sequence number (seq)
        nodes_map = nodes.reset_index()[['id']]
        nodes_map.index.name = 'seq'
        nodes_map = nodes_map.reset_index().set_index('id')
        self._nodes_map = nodes_map

        # Needed to associate edges with sequential ids of from and to nodes
        edges_map = edges
        edges_map['from_seq'] = edges_map['from'].apply(lambda x: nodes_map.loc[x])
        edges_map['to_seq'] = edges_map['to'].apply(lambda x: nodes_map.loc[x])
        edges_map = edges_map[['from_seq', 'to_seq', 'length', 'speed', 'lanes']]

        # Compute sparse graph
        # Note that from and to are inverted, so that the Johnson's algorithm returns the successor node
        self._graph = coo_array((edges_map['length'] / edges_map['speed'], (edges_map['to_seq'], edges_map['from_seq'])))

        # Run the Johnson's algorith
        _, self._next_node = johnson(self._graph, return_predecessors=True)

        # Compute the next_edge matrix
        dgraph = coo_array((np.arange(edges_map.shape[0]), (edges_map['to_seq'], edges_map['from_seq'])))
        to_coord = np.copy(self._next_node)
        from_coord = np.indices((self._next_node.shape[0], self._next_node.shape[1]))[1]
        mask = (to_coord == -9999)
        to_coord[mask] = from_coord[mask]
        self._next_edge = dgraph.toarray()[to_coord, from_coord]

    def _node_to_seq(self, node):
        if np.isscalar(node):
            return self._nodes_map.loc[node]['seq']
        return self._nodes_map.loc[node]['seq'].values

    def _seq_to_node(self, seq):
        if np.isscalar(seq):
            return self._nodes_map.iloc[[seq]].index.values[0]
        return self._nodes_map.iloc[seq].index.values

    def _seq_to_edge(self, seq):
        if np.isscalar(seq):
            return self._edges.iloc[[seq]].index.values[0]
        return self._edges.iloc[seq].index.values

    def next_edge(self, here, there):
        here_seq = self._node_to_seq(here)
        there_seq = self._node_to_seq(there)
        next_seq = self._next_node[there_seq, here_seq]
        next_edge = self._next_edge[there_seq, here_seq]
        return self._seq_to_edge(next_edge), self._seq_to_node(next_seq)


class Simulator:
    class VehicleStatus(Enum):
        WAITING = auto()
        AT_NODE = auto()
        IN_EDGE = auto()
        ARRIVED = auto()

    END_OF_TIME = 999_999

    def __init__(self, nodes, edges, vehicles):
        self._nodes = nodes
        self._edges = edges
        self._vehicles = vehicles
        nr_vehicles = vehicles.shape[0]

        # We could use self.VehicleStatus as type for status below, but performance would be much worse
        self._vstatus = np.zeros(nr_vehicles,
                                 dtype=[('status', np.uint8),
                                        ('node', np.int64),
                                        ('edge', np.int64),
                                        ('destination', np.int64),
                                        ('timer', np.int64)])
        
        self._vstatus['status'] = self.VehicleStatus.WAITING.value
        self._vstatus['timer'] = vehicles['start'].values
        self._vstatus['node'] = vehicles['from_node'].values
        self._vstatus['destination'] = vehicles['to_node'].values

        self._timer = 0

    def step(self, routing):
        # Get the numeric values of the statuses
        waiting_status = self.VehicleStatus.WAITING.value
        at_node_status = self.VehicleStatus.AT_NODE.value
        in_edge_status = self.VehicleStatus.IN_EDGE.value
        arrived_status = self.VehicleStatus.ARRIVED.value

        # Find vehicles to be updated
        starting = (self._vstatus['status'] == waiting_status) & (self._vstatus['timer'] == self._timer)
        # Update their status to AT_NODE
        if np.any(starting):
            self._vstatus['status'][starting] = at_node_status

        # Find vehicles that arrive at a node
        out_of_edge = (self._vstatus['status'] == in_edge_status) & (self._vstatus['timer'] == self._timer)
        # Update their status to AT_NODE
        if np.any(out_of_edge):
            self._vstatus['status'][out_of_edge] = at_node_status

        # Find vehicles that reached their final destination
        arrived = ((self._vstatus['status'] == at_node_status) & (self._vstatus['timer'] == self._timer) &
                   (self._vstatus['node'] == self._vstatus['destination']))
        # Update their status to ARRIVED
        if np.any(arrived):
            self._vstatus['status'][arrived] = arrived_status
            self._vstatus['timer'][arrived] = Simulator.END_OF_TIME

        # Find vehicles that should enter an edge
        entering_edge = (self._vstatus['status'] == at_node_status) & (self._vstatus['timer'] == self._timer)
        # Update their status to IN_EDGE (and define edge and next node)
        if np.any(entering_edge):
            e, n = routing.next_edge(self._vstatus['node'][entering_edge], self._vstatus['destination'][entering_edge])
            self._vstatus['status'][entering_edge] = in_edge_status
            self._vstatus['edge'][entering_edge] = -1  # TODO: does not work with strings
            self._vstatus['node'][entering_edge] = n
            self._vstatus['timer'][entering_edge] = (self._timer +
                                                     np.ceil(self._edges.loc[e]['length'] / self._edges.loc[e]['speed']))

        # Increment the timer
        self._timer += 1


if __name__ == '__main__':
    import time
    
    # Load data
    print("Loading data...")
    start_load = time.time()
    (nodes, edges, vehicles) = read_legacy('nodes__v1.csv', 'edges__v1.csv',
                                         'platoons_size=20_reduction=0.0__v1_seed_0.parquet')
    edges.drop_duplicates(subset=['from', 'to'], inplace=True)
    vehicles = vehicles.iloc[:10000]

    load_time = time.time() - start_load
    print(f"Data loaded in {load_time:.2f} seconds")
    
    # Initialize routing
    print("Initializing routing...")
    start_routing = time.time()
    routing = Routing(nodes=nodes, edges=edges)
    routing_time = time.time() - start_routing
    print(f"Routing initialized in {routing_time:.2f} seconds")
    
    # Run simulation
    print("Starting simulation...")
    simulator = Simulator(nodes=nodes, edges=edges, vehicles=vehicles)

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
    print(f"Total runtime: {load_time + routing_time + sim_time:.2f} seconds")