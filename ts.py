import numpy as np
import pandas as pd


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
