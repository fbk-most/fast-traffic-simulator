"""Sensibility analysis example using SALib on FTS.

Demonstrates:
1. Importing a network from OSMnx.
2. Running a sensitivity analysis using SALib on FTS.

Usage::
    uv run python examples/sensibility.py
"""


import os

import numpy as np

import plotly.graph_objects as go

from fts import Simulator
from fts.visualization import (
    from_osmnx,
    animate,
    animate_occupancy,
    animate_occupancy_mpl,
    plot_edge_occupancy,
    plot_network,
)

OUT_DIR = "examples/viz_output"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers: run_and_log: run simulation and collect logs
#          corner_nodes: identify corner nodes to use the as origins/destinations for vehicles
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


def corner_nodes(pos_proj, pos_ll, N_CORNER=20, plot=True, out_dir=None):
    """Identify N_CORNER closest nodes to bottom-left and top-right corners.
    
    Args:
        pos_proj: dict mapping node IDs to (x, y) projected coordinates
        pos_ll: dict mapping node IDs to (lon, lat) coordinates
        node_ids: array of all node IDs
        N_CORNER: number of closest nodes to each corner
        plot: whether to generate visualization
        out_dir: directory to save HTML plot; if None, only show()
        
    Returns:
        (nodes_bl, nodes_tr): arrays of node IDs closest to corners
    """
    node_ids = np.array(list(pos_proj.keys()))

    coords = np.array([pos_proj[n] for n in node_ids])
    xs, ys = coords[:, 0], coords[:, 1]
    corner_bl = np.array([xs.min(), ys.min()])
    corner_tr = np.array([xs.max(), ys.max()])

    dist_bl = np.linalg.norm(coords - corner_bl, axis=1)
    dist_tr = np.linalg.norm(coords - corner_tr, axis=1)

    idx_bl = np.argsort(dist_bl)[:N_CORNER]
    idx_tr = np.argsort(dist_tr)[:N_CORNER]

    nodes_bl = node_ids[idx_bl]
    nodes_tr = node_ids[idx_tr]

    print(f"Bottom-left corner nodes ({N_CORNER}): {nodes_bl.tolist()}")
    print(f"Top-right corner nodes ({N_CORNER}):   {nodes_tr.tolist()}")

    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scattermap(
            lat=[pos_ll[n][1] for n in node_ids],
            lon=[pos_ll[n][0] for n in node_ids],
            mode="markers",
            marker=dict(size=5, color="lightgrey"),
            name="All nodes",
            hovertext=[str(n) for n in node_ids],
        ))
        fig.add_trace(go.Scattermap(
            lat=[pos_ll[n][1] for n in nodes_bl],
            lon=[pos_ll[n][0] for n in nodes_bl],
            mode="markers+text",
            marker=dict(size=10, color="royalblue"),
            text=[str(n) for n in nodes_bl],
            textposition="top right",
            name="Bottom-left 20",
        ))
        fig.add_trace(go.Scattermap(
            lat=[pos_ll[n][1] for n in nodes_tr],
            lon=[pos_ll[n][0] for n in nodes_tr],
            mode="markers+text",
            marker=dict(size=10, color="crimson"),
            text=[str(n) for n in nodes_tr],
            textposition="top right",
            name="Top-right 20",
        ))
        fig.update_layout(
            map=dict(style="open-street-map", zoom=13, center=dict(lat=46.130, lon=11.247)),
            margin=dict(l=0, r=0, t=30, b=0),
            title="20 closest nodes to bottom-left (blue) and top-right (red) corners",
            legend=dict(x=0, y=1),
        )
        if out_dir:
            fig.write_html(f"{out_dir}/corner_nodes.html")
        fig.show()

    return nodes_bl, nodes_tr    

# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Load network from OSMnx
# ═══════════════════════════════════════════════════════════════════════════

# Download a small neighbourhood
BBOX = (11.227499490547554, 46.11477985577746, 11.26659598835508, 46.145001069749725)
# bbox format: (west, south, east, north)

result = from_osmnx(bbox=BBOX, network_type="drive", force_connected='strong')
edges_osm = result.edges_df
pos_proj = result.pos_projected
pos_ll = result.pos_latlon
edge_geoms = result.edge_geometries

# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Identify 20 closest nodes to bottom-left and top-right corners
# ═══════════════════════════════════════════════════════════════════════════

nodes_bl, nodes_tr = corner_nodes(
    pos_proj, pos_ll, N_CORNER=20, plot=True, out_dir=None
)