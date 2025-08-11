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
        self._graph = coo_array((edges_map['length'], (edges_map['to_seq'], edges_map['from_seq'])))

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


if __name__ == '__main__':
    pass