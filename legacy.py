import pandas as pd
import numpy as np

def read_legacy(nodes_file, edges_file, platoons_file):
    nodes = pd.read_csv(nodes_file, names=['id', 'x', 'y', 'in-out', 'zone'], index_col='id')
    edges = pd.read_csv(edges_file, names=['id', 'from', 'to', 'length', 'speed', 'lanes', 'zone'], index_col='id')
    platoons = pd.read_parquet(platoons_file)[
        ['platoon_id', 'second', 'from_node_id', 'to_node_id', 'nr_vehicles']].set_index('platoon_id')
    nr_vehicles = platoons['nr_vehicles'].sum()
    platoons['last'] = platoons['nr_vehicles'].cumsum()
    platoons['first'] = platoons['last'] - platoons['nr_vehicles']
    v = np.zeros(nr_vehicles, dtype=[('origin', np.int64), ('destination', np.int64), ('start', np.int64)])
    for _, r in platoons.iterrows():
        v[int(r['first']):int(r['last'])] = (r['from_node_id'], r['to_node_id'], r['second'])
    vehicles = pd.DataFrame(v)
    vehicles.index.name = 'id'
    return nodes, edges, vehicles

def map_nodes(legacy_nodes):
    nodes_map = legacy_nodes.reset_index()[['id']]
    nodes_map.index.name = 'seq'
    nodes_map = nodes_map.reset_index().set_index('id')
    return nodes_map

def convert_edges(legacy_edges, nodes_map):
    map_node = lambda n: nodes_map['seq'][n]
    converted_edges = legacy_edges.copy()
    converted_edges['from'] = converted_edges['from'].apply(map_node)
    converted_edges['to'] = converted_edges['to'].apply(map_node)
    converted_edges = converted_edges[['from', 'to', 'length', 'speed', 'lanes']]
    return converted_edges

def convert_vehicles(legacy_vehicles, nodes_map):
    map_node = lambda n: nodes_map['seq'][n]
    converted_vehicles = legacy_vehicles.copy()
    converted_vehicles['origin'] = converted_vehicles['origin'].apply(map_node)
    converted_vehicles['destination'] = converted_vehicles['destination'].apply(map_node)
    return pd.DataFrame(converted_vehicles)
