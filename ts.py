import numpy as np
import pandas as pd
from scipy.sparse import coo_array
from scipy.sparse.csgraph import johnson


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
        self.nodes = nodes
        self.edges = edges

        # Needed to associate nodes with an incremental sequence number (seq)
        nodes_map = nodes.reset_index()[['id']]
        nodes_map.index.name = 'seq'
        nodes_map = nodes_map.reset_index().set_index('id')
        self.nodes_map = nodes_map

        # Needed to associate edges with sequential ids of from and to nodes
        edges_map = edges
        edges_map['from_seq'] = edges_map['from'].apply(lambda x: nodes_map.loc[x])
        edges_map['to_seq'] = edges_map['to'].apply(lambda x: nodes_map.loc[x])
        edges_map = edges_map[['from_seq', 'to_seq', 'length', 'speed', 'lanes']]

        # Compute sparse graph
        # Note that from and to are inverted, so that the Johnson's algorithm returns the successor node
        self.graph = coo_array((edges_map['length'], (edges_map['to_seq'], edges_map['from_seq'])))

        # Run the Johnson's algorith
        self.dist, self.next = johnson(self.graph, return_predecessors=True)

    def _node_to_seq(self, node):
        return int(self.nodes_map.loc[node]['seq'])

    def _seq_to_node(self, seq):
        return int(self.nodes_map.iloc[[seq]].index[0])

    def next_node(self, here, there):
        here_seq = self._node_to_seq(here)
        there_seq = self._node_to_seq(there)
        next_seq = int(self.next[there_seq, here_seq])
        return self._seq_to_node(next_seq), self.dist[next_seq, here_seq]


if __name__ == '__main__':
    pass