"""Sensitivity analysis of FTS using sensitivity_analysis_framework.py.

Quantity of interest: occupancy (%) of a chosen road edge over time.

    occupancy(t) = vehicles_on_edge(t) / capacity(delta) * 100
    capacity     = floor(edge_length / delta) * lanes

Parameters varied:
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

N_CORNER   = None    # nodes near each corner used as origins / destinations

# Optional: set explicit node lists to use as origins / destinations.
# If None, the N_CORNER closest nodes to each corner are used instead.
NODES_BL: list[int] | None = None  # e.g. [123456, 234567, 345678]
NODES_TR: list[int] | None = None  # e.g. [456789, 567890]

NODES_BL = [4, 126, 271, 261]
NODES_TR = [160, 98, 157]

# Edge index to observe.
EDGE_OF_INTEREST = 49

# Set to True to shuffle vehicle priority at each step (uses rng for reproducibility).
RANDOM_PRIORITY = False

# Time axis: occupancy is recorded at each of these steps.
# Increase MAX_STEPS if many vehicles haven't arrived by step 500.
MAX_STEPS = 500
SERIES_AXIS = np.arange(MAX_STEPS)

# Rerouting policy catalogue (GPF discrete realizations).
# Each entry is an interval in steps; None means no rerouting.
REROUTE_POLICIES = [
    None,   # index 0: no rerouting
    100,    # index 1: reroute every 100 steps (infrequent)
    20,     # index 2: reroute every 20 steps  (frequent)
]
N_REROUTE_POLICIES = len(REROUTE_POLICIES)

# SALib / framework settings
SA_PROBLEM = {
    "num_vars": 3,
    "names": ["demand", "od_seed", "reroute_policy"],
    "bounds": [
        [100, 200],                # demand: total vehicles
        [0, 1000],                 # od_seed: cast to int → selects one OD matrix
        [0, N_REROUTE_POLICIES],   # reroute_policy: continuous proxy → floor → discrete index
    ],
}

N_SAMPLES  = 256    # Saltelli base N; total runs = N*(2*D+2) * n_replicas
N_REPLICAS = 1      # stochastic replicas per parameter set (jittered start times)
SEED       = 42     # seed for Saltelli sampling and replica RNG

ANALYZE_VARIANCE = False  # set to True to compute Sobol indices for variance of the time series, in addition to the mean

RUNS_DIR   = os.path.join(OUT_DIR, "runs")

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


if NODES_BL is not None and NODES_TR is not None:
    nodes_bl = np.array(NODES_BL)
    nodes_tr = np.array(NODES_TR)
    print(f"Using explicit origin nodes  ({len(nodes_bl)}): {nodes_bl.tolist()}")
    print(f"Using explicit dest nodes    ({len(nodes_tr)}): {nodes_tr.tolist()}")
else:
    nodes_bl, nodes_tr = corner_nodes(pos_proj, pos_ll, N_CORNER=N_CORNER, plot=True, out_dir=None)

# ═══════════════════════════════════════════════════════════════════════════
# Part 4 — simulator_fn for the framework
# ═══════════════════════════════════════════════════════════════════════════


n_edges = len(edges_osm)


def simulator_fn_all_edges(
    params: dict,
    series_axis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run FTS and return occupancy (%) of ALL edges at each time step.

    Returns:
        Array of shape ``(len(series_axis), n_edges)`` with occupancy in [0, 100].
    """
    demand   = max(1, int(round(float(params["demand"]))))
    od_seed  = int(round(float(params["od_seed"])))

    policy_idx     = int(min(float(params["reroute_policy"]), N_REROUTE_POLICIES - 1e-9))
    reroute_interval = REROUTE_POLICIES[policy_idx]
    do_reroute       = reroute_interval is not None

    od_rng       = np.random.default_rng(od_seed)
    origins      = od_rng.choice(nodes_bl, size=demand, replace=True)
    destinations = od_rng.choice(nodes_tr, size=demand, replace=True)

    start_times = rng.poisson(lam=5.0, size=demand).astype(np.int32)

    vehicles_df = pd.DataFrame({
        "origin":      origins,
        "destination": destinations,
        "start":       start_times,
    })

    sim, _ = Simulator.build(
        edges=edges_osm,
        vehicles=vehicles_df,
        fix_unreachable=True,
        random=RANDOM_PRIORITY,
        rng=rng if RANDOM_PRIORITY else None,
    )

    capacity = compute_edge_capacity(edges_osm, sim.delta)  # shape (n_edges,)
    capacity = np.maximum(capacity, 1)  # avoid division by zero

    max_step    = int(series_axis[-1]) + 1
    occ_buffer  = np.zeros((max_step, n_edges), dtype=np.float32)

    for t in range(max_step):
        edge_ids = sim.vehicles.edge
        counts = np.bincount(edge_ids[edge_ids >= 0], minlength=n_edges)
        occ_buffer[t] = np.minimum(counts / capacity * 100.0, 100.0)

        nr_active = (
            (sim.vehicles.status == Simulator.VehicleStatus.WAITING.value).sum()
            + (sim.vehicles.status == Simulator.VehicleStatus.AT_NODE.value).sum()
            + (sim.vehicles.status == Simulator.VehicleStatus.IN_EDGE.value).sum()
        )
        if nr_active == 0:
            break

        update_routes = do_reroute and (t % reroute_interval == 0)
        sim.step(update_next_leg=update_routes)

    return occ_buffer[series_axis]


def simulator_fn(
    params: dict,
    series_axis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Wrapper that returns occupancy for EDGE_OF_INTEREST only (1D)."""
    return simulator_fn_all_edges(params, series_axis, rng)[:, EDGE_OF_INTEREST]


# ═══════════════════════════════════════════════════════════════════════════
# Part 5 — Run sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════

# Bundle name encodes bounds so that changing them invalidates the cache.
def _bounds_tag(bounds: list) -> str:
    parts = "_".join(f"{lo}-{hi}" for lo, hi in bounds)
    return parts.replace(".", "p")

BUNDLE_NAME = (
    f"N{N_SAMPLES}_R{N_REPLICAS}_seed{SEED}_steps{MAX_STEPS}"
    f"_b{_bounds_tag(SA_PROBLEM['bounds'])}_alledges"
)


print("\n═══ Running sensitivity analysis ═══")
bundle_file = os.path.join(RUNS_DIR, f"{BUNDLE_NAME}.npz")
cached_param_samples = None
if os.path.exists(bundle_file):
    print(f"Loading cached bundle: {BUNDLE_NAME}")
    cached = np.load(bundle_file)
    all_edges_ts = cached["timeseries"]  # (n_ps, n_replicas, n_steps, n_edges)
    cached_param_samples = cached["param_samples"] if "param_samples" in cached else None
    print(f"  shape: {all_edges_ts.shape}")
else:
    # Run all simulations once, recording every edge.
    from SALib.sample import sobol as sobol_sample
    series_axis = np.asarray(SERIES_AXIS)
    param_samples = sobol_sample.sample(SA_PROBLEM, N_SAMPLES, calc_second_order=True, seed=SEED)
    n_ps = param_samples.shape[0]
    n_points = len(series_axis)
    names = SA_PROBLEM["names"]

    print(f"Simulating {n_ps} x {N_REPLICAS} = {n_ps * N_REPLICAS} runs (all {n_edges} edges) ...")
    all_edges_ts = np.empty((n_ps, N_REPLICAS, n_points, n_edges), dtype=np.float32)
    base_rng = np.random.default_rng(SEED)
    for i in range(n_ps):
        p = {name: param_samples[i, j] for j, name in enumerate(names)}
        for r in range(N_REPLICAS):
            rng = np.random.default_rng(base_rng.integers(0, 2**31))
            all_edges_ts[i, r, :, :] = simulator_fn_all_edges(p, series_axis, rng)
        if (i + 1) % max(1, n_ps // 10) == 0:
            print(f"  {i + 1}/{n_ps}")

    os.makedirs(RUNS_DIR, exist_ok=True)
    np.savez_compressed(bundle_file, timeseries=all_edges_ts, param_samples=param_samples)
    print(f"Saved all-edges bundle: {BUNDLE_NAME}")

# Extract edge of interest and run analysis (no simulation needed)
edge_ts = all_edges_ts[:, :, :, EDGE_OF_INTEREST]
print(f"Analysing edge {EDGE_OF_INTEREST}  (shape {edge_ts.shape})")
results = run_sensitivity_analysis(
    simulator_fn=simulator_fn,
    problem=SA_PROBLEM,
    series_axis=SERIES_AXIS,
    n_samples=N_SAMPLES,
    n_replicas=N_REPLICAS,
    seed=SEED,
    precomputed_timeseries=edge_ts,
    precomputed_param_samples=cached_param_samples,
    analyze_variance=ANALYZE_VARIANCE,
)

print_summary(results)

# ═══════════════════════════════════════════════════════════════════════════
# Part 7 — Launch interactive Dash dashboard
# ═══════════════════════════════════════════════════════════════════════════

app = build_dash_app(results)
print("\nStarting dashboard at http://127.0.0.1:8050/")
app.run(debug=False)