"""Sensitivity analysis of FTS using sensitivity_analysis_framework.py.

Quantity of interest: occupancy (%) of a chosen road edge over time.

    occupancy(t) = vehicles_on_edge(t) / capacity(delta) * 100
    capacity     = floor(edge_length / delta) * lanes

Parameters varied:
    delta           — minimum safe following distance (metres); controls jam density
    demand          — total number of vehicles injected into the network
    od_seed         — integer seed for pairing origins (nodes_bl) with destinations
                      (nodes_tr); each distinct integer produces a different OD matrix
    reroute_policy  — categorical (discrete uniform): controls dynamic re-routing
                        0 → no rerouting (static shortest path, fast)
                        1 → reroute every 100 steps (infrequent)
                        2 → reroute every 20 steps  (frequent, most adaptive)
                      Following the GPF framework (Baroni & Tarantola, 2014), this
                      non-scalar / categorical factor is mapped to a discrete uniform
                      index F ∈ {0, 1, 2} so that Sobol variance-based SA can handle
                      it alongside the continuous scalar parameters.

Replica-level stochasticity (within the framework's n_replicas loop) comes
from Poisson-jittered departure times, so each replica of the same parameter
set produces a slightly different traffic wave.

Usage::
    uv run python examples/sensitivity.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fts import Simulator
from fts.visualization import compute_edge_capacity, from_osmnx
from sensitivity_analysis_framework import (
    run_sensitivity_analysis,
    print_summary,
    build_dash_app,
)

OUT_DIR = "examples/viz_output/michela"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — edit these to change the experiment
# ═══════════════════════════════════════════════════════════════════════════

BBOX = (11.227499490547554, 46.11477985577746, 11.26659598835508, 46.145001069749725)

# Edge index to observe. Set to None to auto-select the edge with the most
# traffic in a reference run, or set to an integer (0-based row index in
# edges_osm) to pick a specific road.
EDGE_OF_INTEREST: int | None = None

# Time axis: occupancy is recorded at each of these steps.
# Increase MAX_STEPS if many vehicles haven't arrived by step 500.
MAX_STEPS = 200
SERIES_AXIS = np.arange(MAX_STEPS)

# Rerouting policy catalogue (GPF discrete realizations).
# Each entry is an interval in steps; None means no rerouting.
REROUTE_POLICIES = [
    None,   # index 0: no rerouting
    100,    # index 1: reroute every 100 steps (infrequent)
    10,     # index 2: reroute every 20 steps  (frequent)
]
N_REROUTE_POLICIES = len(REROUTE_POLICIES)

# SALib / framework settings
SA_PROBLEM = {
    "num_vars": 4,
    "names": ["delta", "demand", "od_seed", "reroute_policy"],
    "bounds": [
        [2.0, 20.0],               # delta (m): reciprocal of max density; 2 m → dense; 20 m → sparse
        [50, 100],                 # demand: total vehicles
        [0, 1000],                 # od_seed: cast to int → selects one OD matrix
        [0, N_REROUTE_POLICIES],   # reroute_policy: continuous proxy → floor → discrete index
    ],
}

N_SAMPLES  = 128    # Saltelli base N; total runs = N*(2*D+2) * n_replicas
N_REPLICAS = 10     # stochastic replicas per parameter set (jittered start times)
N_CORNER   = 20    # nodes near each corner used as origins / destinations

# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — Load network
# ═══════════════════════════════════════════════════════════════════════════

print("Loading OSMnx network…")
result    = from_osmnx(bbox=BBOX, network_type="drive", force_connected="strong")
edges_osm = result.edges_df
pos_proj  = result.pos_projected
pos_ll    = result.pos_latlon

# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — Corner nodes (origins and destinations)
# ═══════════════════════════════════════════════════════════════════════════

def corner_nodes(pos_proj, pos_ll, N_CORNER=20, plot=True, out_dir=None):
    """Return arrays of node IDs closest to the bottom-left and top-right corners."""
    node_ids = np.array(list(pos_proj.keys()))
    coords   = np.array([pos_proj[n] for n in node_ids])
    xs, ys   = coords[:, 0], coords[:, 1]

    dist_bl = np.linalg.norm(coords - np.array([xs.min(), ys.min()]), axis=1)
    dist_tr = np.linalg.norm(coords - np.array([xs.max(), ys.max()]), axis=1)

    nodes_bl = node_ids[np.argsort(dist_bl)[:N_CORNER]]
    nodes_tr = node_ids[np.argsort(dist_tr)[:N_CORNER]]

    print(f"Bottom-left corner nodes ({N_CORNER}): {nodes_bl.tolist()}")
    print(f"Top-right corner nodes   ({N_CORNER}): {nodes_tr.tolist()}")

    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scattermap(
            lat=[pos_ll[n][1] for n in node_ids],
            lon=[pos_ll[n][0] for n in node_ids],
            mode="markers", marker=dict(size=5, color="lightgrey"),
            name="All nodes", hovertext=[str(n) for n in node_ids],
        ))
        fig.add_trace(go.Scattermap(
            lat=[pos_ll[n][1] for n in nodes_bl],
            lon=[pos_ll[n][0] for n in nodes_bl],
            mode="markers+text", marker=dict(size=10, color="royalblue"),
            text=[str(n) for n in nodes_bl], textposition="top right",
            name=f"Bottom-left {N_CORNER}",
        ))
        fig.add_trace(go.Scattermap(
            lat=[pos_ll[n][1] for n in nodes_tr],
            lon=[pos_ll[n][0] for n in nodes_tr],
            mode="markers+text", marker=dict(size=10, color="crimson"),
            text=[str(n) for n in nodes_tr], textposition="top right",
            name=f"Top-right {N_CORNER}",
        ))
        fig.update_layout(
            map=dict(style="carto-positron", zoom=13,
                     center=dict(lat=46.130, lon=11.247)),
            margin=dict(l=0, r=0, t=30, b=0),
            title=f"{N_CORNER} closest nodes to bottom-left (blue) and top-right (red) corners",
            legend=dict(x=0, y=1),
        )
        if out_dir:
            fig.write_html(f"{out_dir}/corner_nodes.html")
        fig.show()

    return nodes_bl, nodes_tr


nodes_bl, nodes_tr = corner_nodes(pos_proj, pos_ll, N_CORNER=N_CORNER, plot=True, out_dir=None)

# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — Select edge of interest (if not set manually)
# ═══════════════════════════════════════════════════════════════════════════

EDGE_OF_INTEREST = 388

# Pre-compute edge properties needed for occupancy
_edge_length = float(edges_osm.iloc[EDGE_OF_INTEREST]["length"])
_edge_lanes  = int(edges_osm.iloc[EDGE_OF_INTEREST]["lanes"])

# ═══════════════════════════════════════════════════════════════════════════
# Part 4 — simulator_fn for the framework
# ═══════════════════════════════════════════════════════════════════════════

# Set to True to shuffle vehicle priority at each step (uses rng for reproducibility).
RANDOM_PRIORITY = False


def simulator_fn(
    params: dict,
    series_axis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run FTS and return occupancy (%) of EDGE_OF_INTEREST at each time step.

    Args:
        params:      dict with keys ``delta``, ``demand``, ``od_seed``,
                     and ``reroute_policy`` (continuous proxy clipped to
                     a discrete index into REROUTE_POLICIES).
        series_axis: 1-D array of integer time steps to record occupancy at.
        rng:         NumPy Generator for replica-level stochasticity
                     (Poisson-jittered departure times and, when
                     ``RANDOM_PRIORITY=True``, vehicle-priority shuffling).

    Returns:
        Array of shape ``(len(series_axis),)`` with occupancy values in [0, 100].
    """
    delta    = float(params["delta"])
    demand   = max(1, int(round(float(params["demand"]))))
    od_seed  = int(round(float(params["od_seed"])))

    # --- Categorical: map continuous proxy → discrete reroute policy index ---
    # The continuous sample lies in [0, N_REROUTE_POLICIES); floor gives the
    # integer index, clamped so boundary values (exactly N) map to last policy.
    policy_idx     = int(min(float(params["reroute_policy"]), N_REROUTE_POLICIES - 1e-9))
    reroute_interval = REROUTE_POLICIES[policy_idx]   # None or int
    do_reroute       = reroute_interval is not None

    # --- Build OD pairs (deterministic for this od_seed) ---
    od_rng       = np.random.default_rng(od_seed)
    origins      = od_rng.choice(nodes_bl, size=demand, replace=True)
    destinations = od_rng.choice(nodes_tr, size=demand, replace=True)

    # --- Replica stochasticity: Poisson-jittered start times ---
    start_times = rng.poisson(lam=5.0, size=demand).astype(np.int32)

    vehicles_df = pd.DataFrame({
        "origin":      origins,
        "destination": destinations,
        "start":       start_times,
    })

    sim, _ = Simulator.build(
        edges=edges_osm,
        vehicles=vehicles_df,
        delta=delta,
        fix_unreachable=True,
        random=RANDOM_PRIORITY,
        rng=rng if RANDOM_PRIORITY else None,
    )

    # capacity from the common function (single source of truth)
    capacity = int(compute_edge_capacity(edges_osm, delta)[EDGE_OF_INTEREST])

    max_step    = int(series_axis[-1]) + 1
    occ_buffer  = np.zeros(max_step, dtype=np.float32)

    for t in range(max_step):
        count = int((sim.vehicles.edge == EDGE_OF_INTEREST).sum())
        occ_buffer[t] = min(count / capacity * 100.0, 100.0)

        nr_active = (
            (sim.vehicles.status == Simulator.VehicleStatus.WAITING.value).sum()
            + (sim.vehicles.status == Simulator.VehicleStatus.AT_NODE.value).sum()
            + (sim.vehicles.status == Simulator.VehicleStatus.IN_EDGE.value).sum()
        )
        if nr_active == 0:
            break  # remaining entries stay 0

        update_routes = do_reroute and (t % reroute_interval == 0)
        sim.step(update_next_leg=update_routes)

    return occ_buffer[series_axis]


# ═══════════════════════════════════════════════════════════════════════════
# Part 5 — Run sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════

print("\n═══ Running sensitivity analysis ═══")
results = run_sensitivity_analysis(
    simulator_fn=simulator_fn,
    problem=SA_PROBLEM,
    series_axis=SERIES_AXIS,
    n_samples=N_SAMPLES,
    n_replicas=N_REPLICAS,
)

print_summary(results)

# ═══════════════════════════════════════════════════════════════════════════
# Part 7 — Launch interactive Dash dashboard
# ═══════════════════════════════════════════════════════════════════════════

app = build_dash_app(results)
print("\nStarting dashboard at http://127.0.0.1:8050/")
app.run(debug=False)