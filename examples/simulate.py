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

CSV files may alternatively carry a header row (detected by an ``id`` first
field), in which case OSM-style column names are also accepted: ``u``/``v``
for the edge endpoints and ``maxspeed_mps`` for the edge speed.

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

def _has_header(csv_file: str) -> bool:
    """Detect a header row by an ``id`` first field on the first line."""
    with open(csv_file) as fh:
        return fh.readline().split(',')[0].strip() == 'id'


def _read_input(
    nodes_file: str,
    edges_file: str,
    platoons_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load nodes, edges, and vehicles from UXSim-format files."""
    if _has_header(nodes_file):
        nodes = pd.read_csv(nodes_file, index_col='id')
    else:
        nodes = pd.read_csv(nodes_file, names=['id', 'x', 'y', 'in-out', 'zone'], index_col='id')

    if _has_header(edges_file):
        edges = pd.read_csv(edges_file, index_col='id').rename(
            columns={'u': 'from', 'v': 'to', 'maxspeed_mps': 'speed'}
        )[['from', 'to', 'length', 'speed', 'lanes']]
    else:
        edges = pd.read_csv(
            edges_file,
            names=['id', 'from', 'to', 'length', 'speed', 'lanes', 'zone'],
            index_col='id',
        )

    platoons = pd.read_parquet(platoons_file)[
        ['platoon_id', 'second', 'from_node_id', 'to_node_id', 'nr_vehicles']
    ].set_index('platoon_id')

    repeats = platoons['nr_vehicles'].to_numpy()
    vehicles = pd.DataFrame({
        'origin': np.repeat(platoons['from_node_id'].to_numpy(), repeats),
        'destination': np.repeat(platoons['to_node_id'].to_numpy(), repeats),
        'start': np.repeat(platoons['second'].to_numpy(), repeats),
    })
    vehicles.index.name = 'id'
    return nodes, edges, vehicles


def _map_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """Build a mapping from legacy node IDs to contiguous zero-based indices."""
    nodes_map = nodes.reset_index()[['id']]
    nodes_map.index.name = 'seq'
    return nodes_map.reset_index().set_index('id')


def _convert_edges(edges: pd.DataFrame, nodes_map: pd.DataFrame) -> pd.DataFrame:
    """Remap legacy node IDs in edges to sequential indices."""
    converted = edges.copy()
    converted['from'] = converted['from'].map(nodes_map['seq'])
    converted['to'] = converted['to'].map(nodes_map['seq'])
    return converted[['from', 'to', 'length', 'speed', 'lanes']]


def _convert_vehicles(vehicles: pd.DataFrame, nodes_map: pd.DataFrame) -> pd.DataFrame:
    """Remap legacy node IDs in vehicles to sequential indices."""
    converted = vehicles.copy()
    converted['origin'] = converted['origin'].map(nodes_map['seq'])
    converted['destination'] = converted['destination'].map(nodes_map['seq'])
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
        '--seed', type=int, metavar='N',
        help="Random seed for -r mode (makes random runs reproducible)",
    )
    parser.add_argument(
        '--arrivals', metavar='FILE',
        help="Write per-vehicle arrival times to FILE (Parquet), e.g. for identity testing",
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
        fix_unreachable=True, random=args.random, seed=args.seed,
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
    # Per-step event batches: (step, vehicle_indices, event_kind, edge_index).
    # Kinds: 0 = waiting_at_origin_node, 1 = entered edge, 2 = trip_end.
    # Kept as compact NumPy arrays: a per-event Python list would need ~280
    # bytes per event (gigabytes for city-scale runs) versus ~26 bytes here.
    trip_logs: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
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
                waiting = (simulator.vehicles.started_now & ~simulator.vehicles.entered_edge_now).nonzero()[0]
                entered = simulator.vehicles.entered_edge_now.nonzero()[0]
                arrived = simulator.vehicles.arrived_now.nonzero()[0]
                if len(waiting) + len(entered) + len(arrived) > 0:
                    v_idx = np.concatenate([waiting, entered, arrived])
                    kind = np.repeat(
                        np.array([0, 1, 2], dtype=np.int8),
                        [len(waiting), len(entered), len(arrived)],
                    )
                    edge_idx = np.full(len(v_idx), -1, dtype=np.int32)
                    edge_idx[len(waiting):len(waiting) + len(entered)] = (
                        simulator.vehicles.edge[entered]
                    )
                    trip_logs.append((abs_step, v_idx, kind, edge_idx))

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
        v_idx = np.concatenate([batch[1] for batch in trip_logs])
        t = np.concatenate([
            np.full(len(batch[1]), batch[0], dtype=np.int64) for batch in trip_logs
        ])
        kind = np.concatenate([batch[2] for batch in trip_logs])
        edge_idx = np.concatenate([batch[3] for batch in trip_logs])

        link = np.empty(len(v_idx), dtype=object)
        link[kind == 0] = 'waiting_at_origin_node'
        link[kind == 1] = edges.index.astype(str).to_numpy()[edge_idx[kind == 1]]
        link[kind == 2] = 'trip_end'

        trips_df = pd.DataFrame({
            'name': vehicles.index.to_numpy()[v_idx],
            'orig': vehicles['origin'].to_numpy()[v_idx],
            'dest': vehicles['destination'].to_numpy()[v_idx],
            't': t,
            'link': link,
        })
        trips_df.to_parquet(args.output, index=False)
        print(f"Per-vehicle trip log written to {args.output}")

    if args.arrivals:
        pd.DataFrame({
            'origin': converted_vehicles['origin'].values,
            'destination': converted_vehicles['destination'].values,
            'start': converted_vehicles['start'].values,
            'arrival_time': simulator.vehicles.arrival_time,
        }).to_parquet(args.arrivals)
        print(f"Per-vehicle arrival times written to {args.arrivals}")

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
