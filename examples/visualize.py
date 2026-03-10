"""Visualization and animation example for FTS.

Demonstrates:
1. Static network plot with hover info
2. Animated vehicle movement (Plotly) with play/pause controls
3. Edge occupancy over time
4. OSMnx real-world network with map-based animation (if osmnx installed)

Usage::

    uv run python examples/visualize.py

Figures are saved as interactive HTML files in ``examples/viz_output/``.
"""

import os

import numpy as np
import pandas as pd
from fts import Simulator
from fts.visualization import (
    animate,
    animate_occupancy,
    animate_occupancy_mpl,
    plot_edge_occupancy,
    plot_network,
)

OUT_DIR = "examples/viz_output"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: run simulation and collect logs
# ---------------------------------------------------------------------------

def run_and_log(simulator, vehicles):
    """Run simulation to completion, return (logs, lane_logs) arrays."""
    step_logs = []
    lane_step_logs = []

    def snapshot():
        log = np.vstack([simulator.vehicles.edge, simulator.vehicles.edge_distance])
        log[1, simulator.vehicles.edge == -1] = 0
        return log.swapaxes(0, 1)

    step_logs.append(snapshot())
    lane_step_logs.append(simulator.vehicles.lane.copy())

    while True:
        simulator.step()
        step_logs.append(snapshot())
        lane_step_logs.append(simulator.vehicles.lane.copy())

        nr_active = (
            (simulator.vehicles.status == Simulator.VehicleStatus.WAITING.value).sum()
            + (simulator.vehicles.status == Simulator.VehicleStatus.AT_NODE.value).sum()
            + (simulator.vehicles.status == Simulator.VehicleStatus.IN_EDGE.value).sum()
        )
        if nr_active == 0:
            break

    return np.array(step_logs), np.array(lane_step_logs)


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Synthetic grid network
# ═══════════════════════════════════════════════════════════════════════════

#   0 ──→ 1 ──→ 2
#   ↑     ↑     ↓
#   3 ←── 4 ←── 5
#   ↓     ↓     ↑
#   6 ──→ 7 ──→ 8

edges = pd.DataFrame({
    "from":   [0, 1, 3, 4, 5, 4, 5, 6, 7, 0, 1, 2, 3, 7, 8],
    "to":     [1, 2, 0, 1, 2, 3, 4, 7, 8, 3, 4, 5, 6, 4, 5],
    "length": [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
    "speed":  [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "lanes":  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
})

pos = {
    0: (0, 2), 1: (1, 2), 2: (2, 2),
    3: (0, 1), 4: (1, 1), 5: (2, 1),
    6: (0, 0), 7: (1, 0), 8: (2, 0),
}

vehicles = pd.DataFrame({
    "origin":      [0, 0, 0, 6, 6, 6, 3, 3, 8, 8]*20,
    "destination": [8, 5, 2, 2, 8, 5, 8, 2, 0, 3]*20,
    "start":       [0, 2, 4, 0, 3, 5, 1, 6, 0, 2]*20,
})

simulator, _ = Simulator.build(edges=edges, vehicles=vehicles)
logs, lane_logs = run_and_log(simulator, vehicles)
print(f"Synthetic grid: {logs.shape[0]} steps, {logs.shape[1]} vehicles")

# Static network
fig_net = plot_network(edges, pos, traffic_rule="right")
fig_net.write_html(os.path.join(OUT_DIR, "network.html"))
print(f"  -> {OUT_DIR}/network.html")

# Plotly animation
fig_anim = animate(edges, logs, pos, play_fps=5, tween=True,
                   lane_logs=lane_logs, traffic_rule="right")
fig_anim.write_html(os.path.join(OUT_DIR, "animation.html"))
print(f"  -> {OUT_DIR}/animation.html")

# Plotly animation — subset of vehicles (first 5 only)
fig_anim_sub = animate(edges, logs, pos, play_fps=5, tween=True,
                       lane_logs=lane_logs, traffic_rule="right",
                       vehicle_ids=[0, 1, 2, 3, 4])
fig_anim_sub.write_html(os.path.join(OUT_DIR, "animation_subset.html"))
print(f"  -> {OUT_DIR}/animation_subset.html")

# Occupancy heatmap animation (green -> yellow -> red)
fig_occ_anim = animate_occupancy(edges, logs, pos, play_fps=5,
                                 capacity="DELTA",
                                 traffic_rule="right")
fig_occ_anim.write_html(os.path.join(OUT_DIR, "occupancy_animation.html"))
print(f"  -> {OUT_DIR}/occupancy_animation.html")

# Matplotlib occupancy animation (mp4 export)
try:
    anim_mpl = animate_occupancy_mpl(edges, logs, pos, play_fps=10,
                                     capacity="DELTA",
                                     traffic_rule="right",
                                     save_path=os.path.join(OUT_DIR, "occupancy.mp4"))
    print(f"  -> {OUT_DIR}/occupancy.mp4")
except Exception as e:
    print(f"  Skipping mp4 export ({e})")

# Edge occupancy
fig_occ = plot_edge_occupancy(logs, edges)
fig_occ.update_layout(title="Edge Occupancy Over Time")
fig_occ.write_html(os.path.join(OUT_DIR, "edge_occupancy.html"))
print(f"  -> {OUT_DIR}/edge_occupancy.html")


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: OSMnx real-world network with map animation
# ═══════════════════════════════════════════════════════════════════════════

try:
    from fts.visualization import from_osmnx

    # Download a small neighbourhood
    result = from_osmnx("Piedmont, California, USA", network_type="drive", force_connected='strong')
    edges_osm = result.edges_df
    pos_proj = result.pos_projected
    pos_ll = result.pos_latlon
    edge_geoms = result.edge_geometries

    print(f"\nOSMnx network: {len(edges_osm)} edges, {len(pos_proj)} nodes")

    # Random vehicles
    n_nodes = len(pos_proj)
    rng = np.random.default_rng(42)
    n_trips = 2000
    origins = rng.integers(0, n_nodes, size=n_trips)
    destinations = rng.integers(0, n_nodes, size=n_trips)
    for i in range(n_trips):
        while destinations[i] == origins[i]:
            destinations[i] = rng.integers(0, n_nodes)

    vehicles_osm = pd.DataFrame({
        "origin": origins,
        "destination": destinations,
        "start": rng.integers(0, 10, size=n_trips),
    })

    sim_osm, fixed = Simulator.build(
        edges=edges_osm, vehicles=vehicles_osm, fix_unreachable=True
    )
    if fixed:
        print(f"  Fixed {len(fixed)} unreachable vehicles")

    osm_logs, _ = run_and_log(sim_osm, vehicles_osm)
    print(f"  Simulation: {osm_logs.shape[0]} steps, {osm_logs.shape[1]} vehicles")

    # Plotly map animation (unified — uses pos_latlon)
    fig_map = animate(
        edges_osm, osm_logs, pos_latlon=pos_ll,
        edge_geometries=edge_geoms,
        play_fps=5,
        vehicle_ids=list(rng.choice(osm_logs.shape[1], size=500, replace=False))
    )
    fig_map.write_html(os.path.join(OUT_DIR, "map_animation.html"))
    print(f"  -> {OUT_DIR}/map_animation.html")

    # Static network plot (projected coordinates)
    fig_net_osm = plot_network(edges_osm, pos_proj)
    fig_net_osm.write_html(os.path.join(OUT_DIR, "map_network.html"))
    print(f"  -> {OUT_DIR}/map_network.html")

    # Occupancy heatmap animation on map (unified — uses pos_latlon)
    fig_occ_map = animate_occupancy(
        edges_osm, osm_logs, pos_latlon=pos_ll,
        capacity="DELTA",
        edge_geometries=edge_geoms,
        play_fps=5,
    )
    fig_occ_map.write_html(os.path.join(OUT_DIR, "map_occupancy_animation.html"))
    print(f"  -> {OUT_DIR}/map_occupancy_animation.html")

    # Matplotlib occupancy on map (mp4 export)
    try:
        anim_map_mpl = animate_occupancy_mpl(
            edges_osm, osm_logs, pos_proj,
            capacity="DELTA",
            edge_geometries={
                i: [(c[0], c[1]) for c in coords]
                for i, coords in result.edge_geometries_projected.items()
            } if result.edge_geometries_projected else None,
            crs=result.crs or "EPSG:3857",
            save_path=os.path.join(OUT_DIR, "map_occupancy.mp4"),
        )
        print(f"  -> {OUT_DIR}/map_occupancy.mp4")
    except Exception as e:
        print(f"  Skipping map occupancy mp4 ({e})")

    # Edge occupancy (top 10 busiest edges)
    fig_occ_osm = plot_edge_occupancy(
        osm_logs, edges_osm,
        edge_indices=list(range(min(10, len(edges_osm))))+[715, 541, 97, 540, 95, 93, 121, 156, 324, 326, 323, 866],
    )
    fig_occ_osm.update_layout(title="OSMnx Network - Edge Occupancy")
    fig_occ_osm.write_html(os.path.join(OUT_DIR, "map_edge_occupancy.html"))
    print(f"  -> {OUT_DIR}/map_edge_occupancy.html")

except ImportError as e:
    print(f"\nSkipping OSMnx example (missing dependency: {e})")
    print("Install with: pip install 'fast-traffic-simulator[map]'")

print(f"\nDone. All outputs in {OUT_DIR}/")
