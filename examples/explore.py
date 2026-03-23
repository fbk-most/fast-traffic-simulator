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
from fts.visualization import animate_occupancy, compute_occupancy, from_osmnx, plot_edge_occupancy, plot_network, animate

OUT_DIR = "examples/viz_output/michela"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — edit these to change the experiment
# ═══════════════════════════════════════════════════════════════════════════

BBOX = (11.227499490547554, 46.11477985577746, 11.26659598835508, 46.145001069749725)

# Time axis: occupancy is recorded at each of these steps.
# Increase MAX_STEPS if many vehicles haven't arrived by step 500.
MAX_STEPS = 500
SERIES_AXIS = np.arange(MAX_STEPS)

ANIM_DELTA          = 5.0    # min following distance (m)
ANIM_DEMAND         = 500     # number of vehicles
ANIM_OD_SEED        = 42     # OD matrix seed
ANIM_REROUTE_POLICY = 0      # 0=none, 1=every-100-steps, 2=every-20-steps
ANIM_MAX_STEPS      = MAX_STEPS

# Set to True to shuffle vehicle priority at each step (uses rng for reproducibility).
RANDOM_PRIORITY = False

# Rerouting policy catalogue (GPF discrete realizations).
# Each entry is an interval in steps; None means no rerouting.
REROUTE_POLICIES = [
    None,   # index 0: no rerouting
    100,    # index 1: reroute every 100 steps (infrequent)
    20,     # index 2: reroute every 20 steps  (frequent)
]

N_CORNER   = 4    # nodes near each corner used as origins / destinations

# Optional: set explicit node lists to use as origins / destinations.
# If None, the N_CORNER closest nodes to each corner are used instead.
NODES_BL: list[int] | None = None  # e.g. [123456, 234567, 345678]
NODES_TR: list[int] | None = None  # e.g. [456789, 567890]

NODES_BL = [4, 126, 271, 261]
NODES_TR = [160, 98, 157]

# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — Load network
# ═══════════════════════════════════════════════════════════════════════════

print("Loading OSMnx network…")
result    = from_osmnx(bbox=BBOX, network_type="drive", force_connected="strong")
edges_osm = result.edges_df
pos_proj  = result.pos_projected
pos_ll    = result.pos_latlon
edge_geoms = result.edge_geometries

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

    return nodes_bl, nodes_tr


if NODES_BL is not None and NODES_TR is not None:
    nodes_bl = np.array(NODES_BL)
    nodes_tr = np.array(NODES_TR)
    print(f"Using explicit origin nodes  ({len(nodes_bl)}): {nodes_bl.tolist()}")
    print(f"Using explicit dest nodes    ({len(nodes_tr)}): {nodes_tr.tolist()}")
else:
    nodes_bl, nodes_tr = corner_nodes(pos_proj, pos_ll, N_CORNER=N_CORNER, plot=True, out_dir=None)

# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — Select edge of interest (if not set manually)
# ═══════════════════════════════════════════════════════════════════════════

# Static network plot on map tiles
fig_net_osm = plot_network(edges_osm, pos_latlon=pos_ll, highlight_nodes=[236,215,148,424,412,436,20,18])
fig_net_osm.write_html(os.path.join(OUT_DIR, "map_network.html"))
print(f"  -> {OUT_DIR}/map_network.html")

# ═══════════════════════════════════════════════════════════════════════════
# Part 6 — Concrete animation for a fixed parameter set
# ═══════════════════════════════════════════════════════════════════════════
# Visualise a single representative run so you can inspect vehicle movement
# on the real OSMnx network.  Edit the parameters below to explore scenarios.

print("\n═══ Building animation ═══")
print(f"  delta={ANIM_DELTA} m, demand={ANIM_DEMAND}, "
      f"od_seed={ANIM_OD_SEED}, "
      f"reroute_policy={ANIM_REROUTE_POLICY} "
      f"(interval={REROUTE_POLICIES[ANIM_REROUTE_POLICY]})")

_anim_od_rng     = np.random.default_rng(ANIM_OD_SEED)
_anim_origins    = _anim_od_rng.choice(nodes_bl, size=ANIM_DEMAND, replace=True)
_anim_dests      = _anim_od_rng.choice(nodes_tr, size=ANIM_DEMAND, replace=True)
_anim_starts     = np.zeros(ANIM_DEMAND, dtype=np.int32)   # deterministic for animation

_anim_vehicles_df = pd.DataFrame({
    "origin":      _anim_origins,
    "destination": _anim_dests,
    "start":       _anim_starts,
})

_anim_sim, _ = Simulator.build(
    edges=edges_osm,
    vehicles=_anim_vehicles_df,
    delta=ANIM_DELTA,
    fix_unreachable=True,
    random=False,
)
_anim_interval = REROUTE_POLICIES[ANIM_REROUTE_POLICY]
_do_reroute    = _anim_interval is not None

_step_logs  = []
_lane_logs  = []

def _snapshot(sim):
    log = np.vstack([sim.vehicles.edge, sim.vehicles.edge_distance])
    log[1, sim.vehicles.edge == -1] = 0
    return log.swapaxes(0, 1)

_step_logs.append(_snapshot(_anim_sim))
_lane_logs.append(_anim_sim.vehicles.lane.copy())

for _t in range(ANIM_MAX_STEPS):
    _nr_active = (
        (_anim_sim.vehicles.status == Simulator.VehicleStatus.WAITING.value).sum()
        + (_anim_sim.vehicles.status == Simulator.VehicleStatus.AT_NODE.value).sum()
        + (_anim_sim.vehicles.status == Simulator.VehicleStatus.IN_EDGE.value).sum()
    )
    if _nr_active == 0:
        break
    _update_routes = _do_reroute and (_t % _anim_interval == 0)
    _anim_sim.step(update_next_leg=_update_routes)
    _step_logs.append(_snapshot(_anim_sim))
    _lane_logs.append(_anim_sim.vehicles.lane.copy())

_logs      = np.array(_step_logs)
_lane_logs = np.array(_lane_logs)
print(f"  Collected {_logs.shape[0]} steps for {_logs.shape[1]} vehicles")

# ═══════════════════════════════════════════════════════════════════════════
# Part 7 — Visualize the animation and occupancy
# ═══════════════════════════════════════════════════════════════════════════

_anim_fig = animate(
    edges_osm,
    _logs,
    pos_latlon=pos_ll,
    lane_logs=_lane_logs,
    play_fps=10,
    tween=True,
    edge_geometries=edge_geoms,
)
_anim_path = f"{OUT_DIR}/animation_delta{ANIM_DELTA}_demand{ANIM_DEMAND}_policy{ANIM_REROUTE_POLICY}.html"
_anim_fig.write_html(_anim_path)
print(f"  Animation saved → {_anim_path}")

# Compute occupancy once, use everywhere
_occupancy = compute_occupancy(_logs, edges_osm, delta=ANIM_DELTA)

# Edge occupancy line plot
fig_occ_osm = plot_edge_occupancy(
    _occupancy, edges_osm,
    edge_indices=list(range(len(edges_osm))),
)
fig_occ_osm.update_layout(title="OSMnx Network - Edge Occupancy")
fig_occ_osm.write_html(os.path.join(OUT_DIR, "map_edge_occupancy.html"))
print(f"  -> {OUT_DIR}/map_edge_occupancy.html")

# Occupancy heatmap animation on map
fig_occ_map = animate_occupancy(
    edges_osm, _occupancy, pos_latlon=pos_ll,
    edge_geometries=edge_geoms,
    play_fps=10,
    max_frames=len(_logs),
)
fig_occ_map.write_html(os.path.join(OUT_DIR, "map_occupancy_animation.html"))
print(f"  -> {OUT_DIR}/map_occupancy_animation.html")