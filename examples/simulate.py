"""Simulate traffic on a road network from UXSim-format input files.

Input format
------------
This script reads the input format used by the
`UXSim <https://github.com/toruseo/UXSim>`_ traffic simulator:

- **Nodes CSV** — columns: ``id``, ``x``, ``y``, ``in-out``, ``zone``
- **Edges CSV** — columns: ``id``, ``from``, ``to``, ``length``, ``speed``,
  ``lanes``, ``zone``
- **Platoons Parquet** — columns: ``platoon_id``, ``second``,
  ``from_node_id``, ``to_node_id``, ``nr_vehicles``

Usage
-----
::

    uv run python examples/simulate.py NODES EDGES PLATOONS [options]

Examples::

    # Basic run; writes traffic_volume.parquet in the current directory
    uv run python examples/simulate.py data/nodes.csv data/edges.csv data/platoons.parquet

    # Also write a per-vehicle trip log
    uv run python examples/simulate.py data/nodes.csv data/edges.csv data/platoons.parquet \\
        -o trips.parquet

    # Verbose diagnostics, random ordering, custom SP recomputation interval
    uv run python examples/simulate.py data/nodes.csv data/edges.csv data/platoons.parquet \\
        -v -r --sp 600
"""

import argparse
import time

import numpy as np
import pandas as pd

from fts import Simulator


# ---------------------------------------------------------------------------
# UXSim format I/O helpers
# ---------------------------------------------------------------------------

def _read_input(
    nodes_file: str,
    edges_file: str,
    platoons_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load nodes, edges, and vehicles from UXSim-format files."""
    nodes = pd.read_csv(nodes_file, names=['id', 'x', 'y', 'in-out', 'zone'], index_col='id')
    edges = pd.read_csv(
        edges_file,
        names=['id', 'from', 'to', 'length', 'speed', 'lanes', 'zone'],
        index_col='id',
    )
    platoons = pd.read_parquet(platoons_file)[
        ['platoon_id', 'second', 'from_node_id', 'to_node_id', 'nr_vehicles']
    ].set_index('platoon_id')

    nr_vehicles = platoons['nr_vehicles'].sum()
    platoons['last'] = platoons['nr_vehicles'].cumsum()
    platoons['first'] = platoons['last'] - platoons['nr_vehicles']

    v = np.zeros(nr_vehicles, dtype=[('origin', np.int64), ('destination', np.int64), ('start', np.int64)])
    for _, row in platoons.iterrows():
        v[int(row['first']):int(row['last'])] = (row['from_node_id'], row['to_node_id'], row['second'])

    vehicles = pd.DataFrame(v)
    vehicles.index.name = 'id'
    return nodes, edges, vehicles


def _map_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """Build a mapping from legacy node IDs to contiguous zero-based indices."""
    nodes_map = nodes.reset_index()[['id']]
    nodes_map.index.name = 'seq'
    return nodes_map.reset_index().set_index('id')


def _convert_edges(edges: pd.DataFrame, nodes_map: pd.DataFrame) -> pd.DataFrame:
    """Remap legacy node IDs in edges to sequential indices."""
    map_node = lambda n: nodes_map['seq'][n]
    converted = edges.copy()
    converted['from'] = converted['from'].apply(map_node)
    converted['to'] = converted['to'].apply(map_node)
    return converted[['from', 'to', 'length', 'speed', 'lanes']]


def _convert_vehicles(vehicles: pd.DataFrame, nodes_map: pd.DataFrame) -> pd.DataFrame:
    """Remap legacy node IDs in vehicles to sequential indices."""
    map_node = lambda n: nodes_map['seq'][n]
    converted = vehicles.copy()
    converted['origin'] = converted['origin'].apply(map_node)
    converted['destination'] = converted['destination'].apply(map_node)
    return pd.DataFrame(converted)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate traffic on a road network from UXSim-format input files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('nodes',    help="Nodes CSV file")
    parser.add_argument('edges',    help="Edges CSV file")
    parser.add_argument('platoons', help="Platoons Parquet file")
    parser.add_argument(
        '-o', '--output', metavar='FILE',
        help="Write per-vehicle trip log to FILE (Parquet)",
    )
    parser.add_argument(
        '-t', '--traffic-volume', metavar='FILE',
        help="Write aggregated traffic volume to FILE (Parquet)",
    )
    parser.add_argument(
        '-r', '--random', action='store_true',
        help="Randomise vehicle ordering at nodes (default: deterministic)",
    )
    parser.add_argument(
        '--sp', type=int, default=300, metavar='N',
        help="Shortest-path recomputation interval in steps",
    )
    parser.add_argument(
        '--max-vehicles', type=int, metavar='N',
        help="Limit simulation to the first N vehicles",
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Print verbose diagnostics (duplicate edges, rerouted vehicles)",
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    start_load = time.time()
    nodes, edges, vehicles = _read_input(args.nodes, args.edges, args.platoons)

    # TODO: Manage multi-edges
    n_duplicated = edges.duplicated(subset=['from', 'to']).sum()
    if n_duplicated > 0:
        print(f"*** Removing {n_duplicated} multiple edges ***")
        if args.verbose:
            duplicated = edges[edges.duplicated(subset=['from', 'to'], keep=False)].groupby(['from', 'to'])
            for d, _ in duplicated:
                print("  * Edge cluster: [", ", ".join([str(e) for e in d]), "]")
        edges.drop_duplicates(subset=['from', 'to'], inplace=True)

    if args.max_vehicles:
        vehicles = vehicles.head(args.max_vehicles)

    load_time = time.time() - start_load
    print(f"Data loaded in {load_time:.2f} seconds")

    # Convert data
    print("Converting data...")
    start_convert = time.time()
    nodes_map = _map_nodes(nodes)
    converted_edges = _convert_edges(edges, nodes_map)
    converted_vehicles = _convert_vehicles(vehicles, nodes_map)
    convert_time = time.time() - start_convert
    print(f"Data converted in {convert_time:.2f} seconds")

    # Initialise the simulator
    print("Initializing simulator...")
    start_init = time.time()
    simulator, fixed = Simulator.build(
        edges=converted_edges, vehicles=converted_vehicles,
        fix_unreachable=True, random=args.random,
    )
    if fixed:
        print(f"*** Rerouting {len(fixed)} vehicles with unreachable destinations ***")
        if args.verbose:
            for v in fixed:
                print("  * Vehicle:",
                      f"{vehicles.index[v]}: {vehicles.iloc[v]['origin']} -> {vehicles.iloc[v]['destination']}")
    init_time = time.time() - start_init
    print(f"Simulator initialized in {init_time:.2f} seconds")

    # Run simulation
    print(f"Starting simulation (random={args.random}, SP recomputation every {args.sp} steps)...")
    start_sim = time.time()
    SP = args.sp
    volume_logs: list[pd.DataFrame] = []
    trip_logs: list[list] = []
    record_trips = args.output is not None
    record_volume = args.traffic_volume is not None

    h = 0
    while True:
        for s in range(60 * 60):
            simulator.step(s % SP == SP - 1)
            abs_step = h * 3600 + s

            if record_volume and s % 300 == 299:
                volume_logs.append(pd.DataFrame({
                    'link': edges.index,
                    'time': (abs_step + 1) % (24 * 60 * 60),
                    'traffic_volume': simulator.edges.nr_vehicles,
                }))
                simulator.edges.nr_vehicles = np.zeros_like(simulator.edges.nr_vehicles)

            if record_trips:
                for v in (simulator.vehicles.started_now & ~simulator.vehicles.entered_edge_now).nonzero()[0]:
                    trip_logs.append([vehicles.index[v], vehicles.iloc[v]['origin'],
                                      vehicles.iloc[v]['destination'], abs_step,
                                      "waiting_at_origin_node"])
                for v in simulator.vehicles.entered_edge_now.nonzero()[0]:
                    trip_logs.append([vehicles.index[v], vehicles.iloc[v]['origin'],
                                      vehicles.iloc[v]['destination'], abs_step,
                                      edges.index[simulator.vehicles.edge[v]]])
                for v in simulator.vehicles.arrived_now.nonzero()[0]:
                    trip_logs.append([vehicles.index[v], vehicles.iloc[v]['origin'],
                                      vehicles.iloc[v]['destination'], abs_step,
                                      "trip_end"])

        h += 1
        nr_waiting = (simulator.vehicles.status == Simulator.VehicleStatus.WAITING.value).sum()
        nr_at_node = (simulator.vehicles.status == Simulator.VehicleStatus.AT_NODE.value).sum()
        nr_in_edge = (simulator.vehicles.status == Simulator.VehicleStatus.IN_EDGE.value).sum()
        nr_arrived = (simulator.vehicles.status == Simulator.VehicleStatus.ARRIVED.value).sum()
        print(f"... simulation time after {h} hours: {time.time() - start_sim:.2f} seconds")
        print(f"...... waiting: {nr_waiting} vehicles")
        print(f"...... at node: {nr_at_node} vehicles")
        print(f"...... in edge: {nr_in_edge} vehicles")
        print(f"...... arrived: {nr_arrived} vehicles")

        if nr_waiting + nr_at_node + nr_in_edge == 0:
            break

    # Write outputs
    if record_volume:
        volume = pd.concat(volume_logs).groupby(['link', 'time']).sum().reset_index()
        volume.to_parquet(args.traffic_volume)
        print(f"Traffic volume written to {args.traffic_volume}")

    if record_trips:
        trips_df = pd.DataFrame(trip_logs, columns=['name', 'orig', 'dest', 't', 'link'])
        trips_df.to_parquet(args.output, index=False)
        print(f"Per-vehicle trip log written to {args.output}")

    sim_time = time.time() - start_sim
    total_steps = h * 3600
    print("\n--- Performance Summary ---")
    print(f"Total simulation time: {sim_time:.2f} seconds")
    print(f"Steps per second: {total_steps / sim_time:.2f}")
    print(f"Time per step: {(sim_time / total_steps) * 1000:.4f} ms")
    print(f"Total runtime: {load_time + convert_time + init_time + sim_time:.2f} seconds")

    travel_times = simulator.vehicles.arrival_time - converted_vehicles['start'].values
    print("\n--- Travel Statistics ---")
    print(f"Average travel time: {travel_times.mean():.2f} seconds")
    print(f"Minimum travel time: {travel_times.min():.2f} seconds")
    print(f"Maximum travel time: {travel_times.max():.2f} seconds")


if __name__ == '__main__':
    main()
