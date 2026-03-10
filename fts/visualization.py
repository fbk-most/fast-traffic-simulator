"""Traffic network visualization and animation utilities.

Provides interactive visualizations for FTS simulations:

- :func:`plot_network` — static network diagram with curved edges and hover info.
- :func:`animate` — Plotly animated vehicle movement with play/pause controls.
- :func:`animate_mpl` — Matplotlib animated vehicle movement (faster, supports map tiles).
- :func:`animate_map` — Plotly Scattermap animation on real map tiles.
- :func:`from_osmnx` — import a real road network from OpenStreetMap via OSMnx.
- :func:`animate_occupancy` — animated edge occupancy heatmap (green → red).
- :func:`animate_occupancy_map` — animated edge occupancy on real map tiles.
- :func:`plot_edge_occupancy` — vehicles-per-edge over time.

All functions accept the edges DataFrame used by :class:`fts.Simulator` and a
``pos`` dict mapping node indices to ``(x, y)`` coordinates.  If no positions
are supplied, a spring layout is computed automatically.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import namedtuple
from typing import Dict, Hashable, Literal, Optional, Union, Tuple

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import networkx as nx

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _unit_vectors(
    x0: float, y0: float, x1: float, y1: float, *, traffic_rule: str = "right"
) -> tuple[float, float, float, float, float]:
    """Tangent and normal unit vectors for the segment (x0,y0)->(x1,y1)."""
    dx, dy = float(x1 - x0), float(y1 - y0)
    L = math.hypot(dx, dy)
    if L == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    tx, ty = dx / L, dy / L
    if traffic_rule == "left":
        return tx, ty, -ty, tx, L
    return tx, ty, ty, -tx, L  # right-hand traffic


def _quad_bezier(p0, p1, p2, *, n: int = 24):
    """Sample *n* points along a quadratic Bezier (p0, p1, p2)."""
    ts = np.linspace(0.0, 1.0, n)
    a, b = 1 - ts, ts
    xs = a**2 * p0[0] + 2 * a * b * p1[0] + b**2 * p2[0]
    ys = a**2 * p0[1] + 2 * a * b * p1[1] + b**2 * p2[1]
    return xs.tolist(), ys.tolist()


def _self_loop_points(x0: float, y0: float, *, r: float = 0.08, n: int = 28):
    """Points tracing a circular self-loop near (x0, y0)."""
    cx, cy = x0 + r, y0 + r
    thetas = np.linspace(0.25 * math.pi, 2.25 * math.pi, n)
    return (cx + r * np.cos(thetas)).tolist(), (cy + r * np.sin(thetas)).tolist()


def _cumulative_arc(xs, ys):
    """Cumulative arc-length array for a polyline."""
    xs_a, ys_a = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    diffs = np.hypot(np.diff(xs_a), np.diff(ys_a))
    cum = np.empty(len(xs_a))
    cum[0] = 0.0
    np.cumsum(diffs, out=cum[1:])
    return cum


def _interp_along(xs, ys, cum, target):
    """Interpolate (x, y) at cumulative distance *target* along a polyline."""
    if target <= 0:
        return xs[0], ys[0]
    if target >= cum[-1]:
        return xs[-1], ys[-1]
    j = max(0, min(bisect_right(cum, target) - 1, len(cum) - 2))
    seg = cum[j + 1] - cum[j]
    if seg == 0:
        return xs[j], ys[j]
    r = (target - cum[j]) / seg
    return xs[j] + r * (xs[j + 1] - xs[j]), ys[j] + r * (ys[j + 1] - ys[j])


def _format_attrs(kind: str, label: str, attrs: dict) -> str:
    """Format attributes as Plotly hover HTML."""
    if not attrs:
        return f"<b>{kind}: {label}</b>"
    lines = [f"<b>{kind}: {label}</b>"]
    for k in sorted(attrs, key=str):
        lines.append(f"{k}: {attrs[k]}")
    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# Auto-layout (simple spring / circular fallback)
# ---------------------------------------------------------------------------

def _auto_layout(edges_df) -> Dict[int, tuple]:
    """Compute a spring layout for nodes in *edges_df*."""
    try:
        import networkx as nx
        G = nx.DiGraph()
        for _, row in edges_df.iterrows():
            G.add_edge(int(row["from"]), int(row["to"]))
        return nx.spring_layout(G, seed=42)
    except ImportError:
        nodes = sorted(
            set(edges_df["from"].astype(int)) | set(edges_df["to"].astype(int))
        )
        n = len(nodes)
        return {
            nd: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
            for i, nd in enumerate(nodes)
        }


# ---------------------------------------------------------------------------
# Edge geometry builder (shared by plot_network and animate)
# ---------------------------------------------------------------------------

def _build_edge_geometry(
    edges_df,
    pos: Dict[Hashable, tuple],
    *,
    edge_curvature: float = 0.22,
    base_offset: float = 0.55,
    parallel_spacing: float = 0.35,
    parallel_exponent: float = 1.6,
    traffic_rule: str = "right",
    curve_single_edges: bool = False,
    curve_samples: int = 48,
):
    """Pre-compute Bezier paths, cumulative lengths, and metadata for every edge.

    Returns a list of dicts, one per edge row, each containing:
    ``xs``, ``ys``, ``cum``, ``total``, ``u``, ``v``, ``length_attr``, ``lanes``.
    """
    from_nodes = edges_df["from"].values.astype(int)
    to_nodes = edges_df["to"].values.astype(int)

    # Pre-extract columns as numpy arrays (avoids .iloc per iteration)
    lengths = edges_df["length"].values.astype(float) if "length" in edges_df.columns else None
    lanes_col = edges_df["lanes"].values.astype(int) if "lanes" in edges_df.columns else None

    # Group parallel edges: (u, v) -> list of row indices
    by_uv: dict[tuple, list[int]] = {}
    for idx in range(len(from_nodes)):
        by_uv.setdefault((from_nodes[idx], to_nodes[idx]), []).append(idx)

    opp_count = {uv: len(by_uv.get((uv[1], uv[0]), [])) for uv in by_uv}

    results: list[dict] = [None] * len(from_nodes)  # type: ignore[list-item]

    for (u, v), indices in by_uv.items():
        x0, y0 = float(pos[u][0]), float(pos[u][1])
        x2, y2 = float(pos[v][0]), float(pos[v][1])

        if u == v:
            for rank, idx in enumerate(indices):
                r = edge_curvature * (0.4 + 0.18 * rank) + 0.02
                xs, ys = _self_loop_points(x0, y0, r=r, n=28)
                cum = _cumulative_arc(xs, ys)
                length_val = float(lengths[idx]) if lengths is not None and cum[-1] > 0 else float(cum[-1]) if cum[-1] > 0 else 1.0
                lanes_val = max(1, int(lanes_col[idx])) if lanes_col is not None else 1
                results[idx] = dict(
                    xs=np.asarray(xs), ys=np.asarray(ys), cum=cum, total=float(cum[-1]),
                    u=u, v=v, length_attr=length_val, lanes=lanes_val,
                )
            continue

        tx, ty, nx_, ny_, L = _unit_vectors(x0, y0, x2, y2, traffic_rule=traffic_rule)
        mx, my = (x0 + x2) / 2.0, (y0 + y2) / 2.0
        bow = edge_curvature * L
        m = len(indices)

        if m == 1:
            should_curve = curve_single_edges or opp_count.get((u, v), 0) > 0
            offsets = [base_offset if should_curve else 0.0]
        else:
            start = -(m - 1) / 2.0
            offsets = [
                base_offset
                + math.copysign(
                    ((abs(start + i) + 1.0) ** parallel_exponent - 1.0) * parallel_spacing,
                    start + i,
                )
                for i in range(m)
            ]

        for rank, idx in enumerate(indices):
            cx = mx + offsets[rank] * bow * nx_
            cy = my + offsets[rank] * bow * ny_
            xs, ys = _quad_bezier((x0, y0), (cx, cy), (x2, y2), n=curve_samples)
            cum = _cumulative_arc(xs, ys)
            length_val = float(lengths[idx]) if lengths is not None and cum[-1] > 0 else float(cum[-1]) if cum[-1] > 0 else 1.0
            lanes_val = max(1, int(lanes_col[idx])) if lanes_col is not None else 1
            results[idx] = dict(
                xs=np.asarray(xs), ys=np.asarray(ys), cum=cum, total=float(cum[-1]),
                u=u, v=v, length_attr=length_val, lanes=lanes_val,
            )

    return results


# ---------------------------------------------------------------------------
# Vectorized position precomputation (shared by animate, animate_mpl)
# ---------------------------------------------------------------------------

def _precompute_positions(
    logs: np.ndarray,
    geom: list[dict],
    *,
    lane_logs: Optional[np.ndarray] = None,
    lane_offset: float = 0.06,
    traffic_rule: str = "right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (X, Y, OP) arrays for all steps and vehicles.

    Returns arrays of shape ``(n_steps, n_veh)``.
    """
    n_steps, n_veh, _ = logs.shape

    # Pre-extract geometry arrays
    n_edges = len(geom)
    geom_xs = [np.asarray(g["xs"], dtype=float) for g in geom]
    geom_ys = [np.asarray(g["ys"], dtype=float) for g in geom]
    geom_cum = [np.asarray(g["cum"], dtype=float) for g in geom]
    geom_total = np.array([g["total"] for g in geom], dtype=float)
    geom_length = np.array([g["length_attr"] for g in geom], dtype=float)
    geom_lanes = np.array([g["lanes"] for g in geom], dtype=int)

    X = np.full((n_steps, n_veh), np.nan)
    Y = np.full((n_steps, n_veh), np.nan)
    OP = np.zeros((n_steps, n_veh))

    # Flatten active entries
    e_idx_all = logs[:, :, 0].astype(int)
    e_dist_all = logs[:, :, 1].astype(float)
    active_mask = e_idx_all >= 0
    flat_t, flat_v = np.nonzero(active_mask)

    if len(flat_t) == 0:
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)
        return X, Y, OP

    flat_eidx = e_idx_all[flat_t, flat_v]
    flat_edist = e_dist_all[flat_t, flat_v]

    # Allocate flat result arrays
    flat_x = np.empty(len(flat_t))
    flat_y = np.empty(len(flat_t))

    # Group by edge and batch-interpolate
    for eidx in range(n_edges):
        sel = flat_eidx == eidx
        if not np.any(sel):
            continue

        dists = flat_edist[sel]
        L_attr = geom_length[eidx]
        total = geom_total[eidx]
        xs_e = geom_xs[eidx]
        ys_e = geom_ys[eidx]
        cum_e = geom_cum[eidx]

        # Compute interpolation targets
        ratios = np.clip(dists / L_attr, 0.0, 1.0) if L_attr > 0 else np.zeros_like(dists)
        targets = ratios * total if total > 0 else np.zeros_like(ratios)

        # Vectorized searchsorted + linear interpolation
        j = np.clip(np.searchsorted(cum_e, targets, side="right") - 1, 0, len(cum_e) - 2)
        seg = cum_e[j + 1] - cum_e[j]
        safe_seg = np.where(seg == 0, 1.0, seg)
        r = np.where(seg == 0, 0.0, (targets - cum_e[j]) / safe_seg)
        r = np.clip(r, 0.0, 1.0)

        xv = xs_e[j] + r * (xs_e[j + 1] - xs_e[j])
        yv = ys_e[j] + r * (ys_e[j + 1] - ys_e[j])

        # Lane offsets (vectorized)
        if lane_logs is not None and geom_lanes[eidx] > 1:
            t_sel = flat_t[sel]
            v_sel = flat_v[sel]
            lane_idx = lane_logs[t_sel, v_sel].astype(int)
            valid_lane = lane_idx >= 0
            if np.any(valid_lane):
                n_lanes = geom_lanes[eidx]
                centered = lane_idx[valid_lane] - (n_lanes - 1) / 2.0
                if traffic_rule == "left":
                    centered = -centered
                # Local normals at interpolation points
                dx_seg = xs_e[j[valid_lane] + 1] - xs_e[j[valid_lane]]
                dy_seg = ys_e[j[valid_lane] + 1] - ys_e[j[valid_lane]]
                norm = np.hypot(dx_seg, dy_seg)
                norm = np.where(norm < 1e-12, 1.0, norm)
                nx_ = -dy_seg / norm
                ny_ = dx_seg / norm
                xv[valid_lane] += centered * lane_offset * nx_
                yv[valid_lane] += centered * lane_offset * ny_

        flat_x[sel] = xv
        flat_y[sel] = yv

    X[flat_t, flat_v] = flat_x
    Y[flat_t, flat_v] = flat_y
    OP[flat_t, flat_v] = 1.0

    # Forward-fill NaN positions (vectorized via pandas)
    # ffill: after arrival, position stays at last known location
    # bfill: before first appearance, position is at spawn point (prevents jump)
    X = pd.DataFrame(X).ffill().bfill().fillna(0.0).values
    Y = pd.DataFrame(Y).ffill().bfill().fillna(0.0).values

    return X, Y, OP


# ---------------------------------------------------------------------------
# Vehicle colour helper
# ---------------------------------------------------------------------------

_PALETTES = (
    pc.qualitative.Dark24 + pc.qualitative.Set3
    + pc.qualitative.Alphabet + pc.qualitative.Safe
)


def _vehicle_colors_and_hover(
    logs: np.ndarray, n_veh: int, vehicle_colors: Optional[Dict[int, str]]
):
    """Return (colors_list, hover_list) for all vehicles."""
    if vehicle_colors is not None:
        return (
            [vehicle_colors.get(i, "#888") for i in range(n_veh)],
            [f"veh {i}" for i in range(n_veh)],
        )
    od_pairs = []
    for v in range(n_veh):
        edges_v = logs[:, v, 0]
        active = edges_v[edges_v >= 0]
        if len(active) > 0:
            od_pairs.append((int(active[0]), int(active[-1])))
        else:
            od_pairs.append((-1, -1))
    unique_ods = list(dict.fromkeys(od_pairs))
    od2col = {od: _PALETTES[i % len(_PALETTES)] for i, od in enumerate(unique_ods)}
    colors = [od2col[od] for od in od_pairs]
    hover = [
        f"veh {i} | edges {o}\u2192{d}" if o >= 0 else f"veh {i}"
        for i, (o, d) in enumerate(od_pairs)
    ]
    return colors, hover


# ---------------------------------------------------------------------------
# plot_network
# ---------------------------------------------------------------------------

def plot_network(
    edges_df,
    pos: Optional[Dict[Hashable, tuple]] = None,
    *,
    node_size: int = 12,
    edge_width: float = 1.8,
    edge_color: str = "#888",
    edge_curvature: float = 0.22,
    base_offset: float = 0.55,
    parallel_spacing: float = 0.35,
    parallel_exponent: float = 1.6,
    traffic_rule: Literal["right", "left"] = "right",
    curve_single_edges: bool = False,
) -> go.Figure:
    """Plot the traffic network as an interactive Plotly figure.

    Parameters
    ----------
    edges_df : DataFrame
        Edges table with at least ``from``, ``to``, ``length``, ``speed``,
        ``lanes`` columns (same format accepted by :class:`fts.Simulator`).
    pos : dict or None
        Mapping ``{node_id: (x, y)}``.  If *None*, a spring layout is used.
    node_size, edge_width, edge_color
        Visual styling for nodes and edges.
    edge_curvature, base_offset, parallel_spacing, parallel_exponent
        Control the curvature of edges and spacing between parallel edges.
    traffic_rule : ``"right"`` or ``"left"``
        Which side of the road vehicles travel on (affects curve direction).
    curve_single_edges : bool
        Whether to curve edges that have no parallel counterpart.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if pos is None:
        pos = _auto_layout(edges_df)

    geom = _build_edge_geometry(
        edges_df, pos,
        edge_curvature=edge_curvature, base_offset=base_offset,
        parallel_spacing=parallel_spacing, parallel_exponent=parallel_exponent,
        traffic_rule=traffic_rule, curve_single_edges=curve_single_edges,
        curve_samples=24,
    )

    # --- single combined edge trace (None separators) ---
    all_ex: list = []
    all_ey: list = []
    mid_x, mid_y, mid_text = [], [], []

    for idx, g in enumerate(geom):
        xs, ys = g["xs"].tolist(), g["ys"].tolist()
        all_ex.extend(xs)
        all_ex.append(None)
        all_ey.extend(ys)
        all_ey.append(None)
        mi = len(xs) // 2
        mid_x.append(xs[mi])
        mid_y.append(ys[mi])
        row = edges_df.iloc[idx]
        attrs = {c: row[c] for c in edges_df.columns if c not in ("from", "to")}
        mid_text.append(_format_attrs("edge", f"{g['u']} \u2192 {g['v']} (idx={idx})", attrs))

    edge_trace = go.Scatter(
        x=all_ex, y=all_ey, mode="lines",
        line=dict(width=edge_width, color=edge_color),
        hoverinfo="skip", showlegend=False,
    )

    edge_hover = go.Scatter(
        x=mid_x, y=mid_y, mode="markers",
        marker=dict(size=10, opacity=0),
        hovertemplate="%{text}<extra></extra>", text=mid_text,
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font=dict(color="black")),
    )

    # --- node trace ---
    nodes = sorted(set(edges_df["from"].astype(int)) | set(edges_df["to"].astype(int)))
    xN = [float(pos[n][0]) for n in nodes]
    yN = [float(pos[n][1]) for n in nodes]
    node_text = [f"<b>node {n}</b>" for n in nodes]

    node_trace = go.Scatter(
        x=xN, y=yN, mode="markers",
        marker=dict(size=node_size, color="black"),
        hovertemplate="%{text}<extra></extra>", text=node_text,
        name="nodes",
    )

    fig = go.Figure(data=[edge_trace, edge_hover, node_trace])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=True),
        yaxis=dict(showgrid=False, zeroline=False, visible=True,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# animate (Plotly)
# ---------------------------------------------------------------------------

def animate(
    edges_df,
    logs: np.ndarray,
    pos: Optional[Dict[Hashable, tuple]] = None,
    *,
    play_fps: int = 5,
    marker_size: int = 8,
    lane_offset: float = 0.06,
    tween: bool = True,
    lane_logs: Optional[np.ndarray] = None,
    vehicle_colors: Optional[Dict[int, str]] = None,
    vehicle_ids: Optional[list[int]] = None,
    edge_curvature: float = 0.22,
    base_offset: float = 0.55,
    parallel_spacing: float = 0.35,
    parallel_exponent: float = 1.6,
    traffic_rule: Literal["right", "left"] = "right",
    curve_single_edges: bool = False,
) -> go.Figure:
    """Animate vehicle movement on the network.

    Parameters
    ----------
    edges_df : DataFrame
        Edges table (same as :func:`plot_network`).
    logs : ndarray of shape ``(steps, n_vehicles, 2)``
        Position log as produced by the demo recording loop.  For each vehicle
        at each step: ``[edge_index, edge_distance]``.  Edge index ``-1``
        means the vehicle is not currently on an edge (waiting/arrived).
    pos : dict or None
        Node positions.  Auto-computed if *None*.
    play_fps : int
        Playback frames per second.
    marker_size : int
        Vehicle marker diameter.
    lane_offset : float
        Lateral offset per lane in plot units.
    tween : bool
        Smooth interpolation between frames.
    lane_logs : ndarray of shape ``(steps, n_vehicles)`` or None
        Per-step lane index for each vehicle.  If *None*, lane offsets are
        not applied.
    vehicle_colors : dict or None
        ``{vehicle_id: color_string}``.  If *None*, vehicles are coloured by
        their origin-destination pair inferred from the logs.
    vehicle_ids : list of int or None
        Subset of vehicle indices to draw.  If *None*, all vehicles are shown.
    edge_curvature, base_offset, parallel_spacing, parallel_exponent,
    traffic_rule, curve_single_edges
        Passed through to :func:`plot_network` / edge geometry builder.

    Returns
    -------
    plotly.graph_objects.Figure
        Animated figure with play/pause button and time slider.
    """
    if logs.ndim != 3 or logs.shape[2] != 2:
        raise ValueError(
            f"logs must have shape (steps, n_vehicles, 2), got {logs.shape}"
        )

    # --- vehicle subset filtering ---
    orig_ids = None
    if vehicle_ids is not None:
        orig_ids = np.asarray(sorted(vehicle_ids), dtype=int)
        logs = logs[:, orig_ids, :]
        if lane_logs is not None:
            lane_logs = lane_logs[:, orig_ids]
        if vehicle_colors is not None:
            vehicle_colors = {
                i: vehicle_colors[oid]
                for i, oid in enumerate(orig_ids)
                if oid in vehicle_colors
            }

    n_steps, n_veh, _ = logs.shape
    if pos is None:
        pos = _auto_layout(edges_df)

    # --- base network figure ---
    base_fig = plot_network(
        edges_df, pos,
        edge_curvature=edge_curvature, base_offset=base_offset,
        parallel_spacing=parallel_spacing, parallel_exponent=parallel_exponent,
        traffic_rule=traffic_rule, curve_single_edges=curve_single_edges,
    )
    fig = go.Figure(data=list(base_fig.data), layout=base_fig.layout)

    if n_veh == 0:
        fig.frames = []
        return fig

    # --- precompute edge geometry (higher resolution for animation) ---
    geom = _build_edge_geometry(
        edges_df, pos,
        edge_curvature=edge_curvature, base_offset=base_offset,
        parallel_spacing=parallel_spacing, parallel_exponent=parallel_exponent,
        traffic_rule=traffic_rule, curve_single_edges=curve_single_edges,
        curve_samples=48,
    )

    # --- colours ---
    colors, hover = _vehicle_colors_and_hover(logs, n_veh, vehicle_colors)
    # Restore original vehicle IDs in hover text
    if orig_ids is not None:
        hover = [h.replace(f"veh {i}", f"veh {orig_ids[i]}") for i, h in enumerate(hover)]

    # --- precompute all positions (vectorized) ---
    X, Y, OP = _precompute_positions(
        logs, geom,
        lane_logs=lane_logs, lane_offset=lane_offset, traffic_rule=traffic_rule,
    )

    hover_arr = np.array(hover, dtype=object)

    # --- initial vehicle trace ---
    text0 = np.where(OP[0] > 0, hover_arr, "")
    fig.add_trace(go.Scatter(
        x=X[0], y=Y[0], mode="markers",
        marker=dict(size=marker_size, symbol="circle",
                    line=dict(color="black", width=1),
                    color=colors, opacity=OP[0]),
        hovertemplate="%{text}<extra></extra>", text=text0,
        showlegend=False,
    ))
    veh_idx = len(fig.data) - 1

    # --- frames ---
    frames = []
    for t in range(n_steps):
        text_t = np.where(OP[t] > 0, hover_arr, "")
        frames.append(go.Frame(
            name=str(t),
            data=[go.Scatter(x=X[t], y=Y[t], marker=dict(opacity=OP[t]), text=text_t)],
            traces=[veh_idx],
        ))
    fig.frames = frames

    # --- axis range ---
    xs_all = [float(pos[n][0]) for n in pos]
    ys_all = [float(pos[n][1]) for n in pos]
    pad_x = (max(xs_all) - min(xs_all)) * 0.05 or 1.0
    pad_y = (max(ys_all) - min(ys_all)) * 0.05 or 1.0
    fig.update_xaxes(range=[min(xs_all) - pad_x, max(xs_all) + pad_x])
    fig.update_yaxes(range=[min(ys_all) - pad_y, max(ys_all) + pad_y],
                     scaleanchor="x", scaleratio=1)

    # --- controls ---
    ms = max(1, int(1000 / max(1, play_fps)))
    trans = {"duration": ms, "easing": "linear"} if tween else {"duration": 0}
    play_args = {"fromcurrent": True, "frame": {"duration": ms, "redraw": False},
                 "transition": trans}

    fig.update_layout(
        margin=dict(b=90),
        sliders=[{
            "active": 0, "y": -0.07, "x": 0.15, "len": 0.72,
            "pad": {"t": 20, "b": 0},
            "currentvalue": {"prefix": "t = ", "visible": True},
            "steps": [
                {"args": [[str(t)], {"mode": "immediate",
                                     "frame": {"duration": 0, "redraw": False},
                                     "transition": {"duration": 0}}],
                 "label": str(t), "method": "animate"}
                for t in range(n_steps)
            ],
        }],
        updatemenus=[{
            "type": "buttons", "direction": "left",
            "x": 0.02, "y": -0.12, "showactive": True,
            "buttons": [{
                "label": "\u25b6 / \u23f8", "method": "animate",
                "args": [None, play_args],
                "args2": [[None], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": False},
                                   "transition": {"duration": 0}}],
            }],
        }],
    )

    return fig


# ---------------------------------------------------------------------------
# animate_mpl (Matplotlib + optional contextily map tiles)
# ---------------------------------------------------------------------------

def animate_mpl(
    edges_df,
    logs: np.ndarray,
    pos: Optional[Dict[Hashable, tuple]] = None,
    *,
    play_fps: int = 10,
    marker_size: int = 20,
    lane_offset: float = 0.06,
    lane_logs: Optional[np.ndarray] = None,
    vehicle_colors: Optional[Dict[int, str]] = None,
    vehicle_ids: Optional[list[int]] = None,
    edge_curvature: float = 0.22,
    base_offset: float = 0.55,
    parallel_spacing: float = 0.35,
    parallel_exponent: float = 1.6,
    traffic_rule: Literal["right", "left"] = "right",
    curve_single_edges: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 100,
    figsize: tuple = (10, 8),
    tile_source=None,
    crs: Optional[str] = None,
):
    """Animate vehicles using Matplotlib FuncAnimation with blit.

    Uses a single ``ax.scatter()`` PathCollection updated in-place each frame
    (faster than individual Circle patches — see plan notes).

    Parameters
    ----------
    edges_df, logs, pos, lane_offset, lane_logs, vehicle_colors,
    edge_curvature, base_offset, parallel_spacing, parallel_exponent,
    traffic_rule, curve_single_edges
        Same as :func:`animate`.
    play_fps : int
        Playback frames per second.
    marker_size : int
        Scatter marker size (points²).
    save_path : str or None
        If given, save the animation (e.g. ``"out.mp4"`` or ``"out.gif"``).
    dpi : int
        Resolution for saved animation.
    figsize : tuple
        Figure size in inches.
    tile_source
        Contextily tile provider (e.g. ``contextily.providers.CartoDB.Positron``).
        Defaults to OpenStreetMap if *crs* is set.
    crs : str or None
        Coordinate reference system of *pos* (e.g. ``"EPSG:4326"``).
        If provided and contextily is available, a map tile background is added.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import LineCollection
    except ImportError:
        raise ImportError("matplotlib is required for animate_mpl(). Install with: pip install matplotlib")

    if logs.ndim != 3 or logs.shape[2] != 2:
        raise ValueError(f"logs must have shape (steps, n_vehicles, 2), got {logs.shape}")

    # --- vehicle subset filtering ---
    if vehicle_ids is not None:
        orig_ids = np.asarray(sorted(vehicle_ids), dtype=int)
        logs = logs[:, orig_ids, :]
        if lane_logs is not None:
            lane_logs = lane_logs[:, orig_ids]
        if vehicle_colors is not None:
            vehicle_colors = {
                i: vehicle_colors[oid]
                for i, oid in enumerate(orig_ids)
                if oid in vehicle_colors
            }

    n_steps, n_veh, _ = logs.shape
    if pos is None:
        pos = _auto_layout(edges_df)

    # --- edge geometry + position precomputation ---
    geom = _build_edge_geometry(
        edges_df, pos,
        edge_curvature=edge_curvature, base_offset=base_offset,
        parallel_spacing=parallel_spacing, parallel_exponent=parallel_exponent,
        traffic_rule=traffic_rule, curve_single_edges=curve_single_edges,
        curve_samples=48,
    )

    X, Y, OP = _precompute_positions(
        logs, geom,
        lane_logs=lane_logs, lane_offset=lane_offset, traffic_rule=traffic_rule,
    )

    colors, _ = _vehicle_colors_and_hover(logs, n_veh, vehicle_colors)

    # --- figure setup ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # Draw edges as LineCollection (single artist, fast)
    segments = []
    for g in geom:
        pts = np.column_stack([g["xs"], g["ys"]])
        segments.append(pts)
    lc = LineCollection(segments, colors="#888", linewidths=1.2)
    ax.add_collection(lc)

    # Draw nodes
    node_x = [float(pos[n][0]) for n in pos]
    node_y = [float(pos[n][1]) for n in pos]
    ax.scatter(node_x, node_y, c="black", s=30, zorder=3)

    # Map tiles
    if crs is not None:
        try:
            import contextily as cx
            src = tile_source or cx.providers.OpenStreetMap.Mapnik
            cx.add_basemap(ax, crs=crs, source=src)
        except ImportError:
            pass  # no contextily, skip tiles

    # Auto-range
    xs_all = [float(pos[n][0]) for n in pos]
    ys_all = [float(pos[n][1]) for n in pos]
    pad_x = (max(xs_all) - min(xs_all)) * 0.05 or 1.0
    pad_y = (max(ys_all) - min(ys_all)) * 0.05 or 1.0
    ax.set_xlim(min(xs_all) - pad_x, max(xs_all) + pad_x)
    ax.set_ylim(min(ys_all) - pad_y, max(ys_all) + pad_y)

    # Vehicle scatter (single PathCollection for all vehicles)
    scat = ax.scatter(
        X[0], Y[0], s=marker_size, c=colors, edgecolors="black",
        linewidths=0.5, zorder=5, alpha=OP[0],
    )

    title_text = ax.set_title("t = 0")

    def _init():
        return (scat, title_text)

    def _update(frame):
        offsets = np.column_stack([X[frame], Y[frame]])
        scat.set_offsets(offsets)
        scat.set_alpha(OP[frame])
        title_text.set_text(f"t = {frame}")
        return (scat, title_text)

    anim = FuncAnimation(
        fig, _update, init_func=_init,
        frames=n_steps, blit=True,
        interval=int(1000 / max(1, play_fps)),
    )

    if save_path is not None:
        ext = save_path.rsplit(".", 1)[-1].lower()
        if ext == "gif":
            anim.save(save_path, writer="pillow", dpi=dpi)
        else:
            anim.save(save_path, writer="ffmpeg", dpi=dpi)

    return anim


# ---------------------------------------------------------------------------
# from_osmnx — OSMnx adapter
# ---------------------------------------------------------------------------

OSMnxResult = namedtuple("OSMnxResult", [
    "edges_df", "pos_projected", "pos_latlon", "edge_geometries",
])


def from_osmnx(
    place_or_graph: Union[str, nx.MultiDiGraph, None] = None,
    *,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    simplify: bool = True,
    force_connected: bool = True,
    network_type: str = "drive",
    default_speed: Optional[float] = None,  # km/h
    default_lanes: Optional[int] = None,
) -> OSMnxResult:
    """Import a road network from OSM via OSMnx with strict NaN prevention."""
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("osmnx is required. Install with: pip install 'fast-traffic-simulator[map]'")

    # 1. Acquire Graph
    if place_or_graph is not None:
        if isinstance(place_or_graph, str):
            G = ox.graph_from_place(place_or_graph, network_type=network_type, simplify=simplify)
        else:
            G = place_or_graph.copy()
            if simplify:
                G = ox.simplify_graph(G)
    elif bbox is not None:
        G = ox.graph_from_bbox(bbox=bbox, network_type=network_type, simplify=simplify)
    else:
        raise ValueError("Provide 'place_or_graph' or 'bbox'.")

    # 2. Force Strong Connectivity (NetworkX)
    if force_connected:
        nodes = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(nodes).copy()

    # 3. Routing Metrics (Adds speed_kph and travel_time)
    # If default_speed is None, OSMnx uses its own internal highway-type defaults.
    G = ox.routing.add_edge_speeds(G, fallback=default_speed)
    G = ox.routing.add_edge_travel_times(G)

    # 4. Projections & Mapping
    G_ll = G
    G_proj = ox.project_graph(G)
    osm_to_fts = {osm_id: i for i, osm_id in enumerate(G_proj.nodes)}

    pos_projected = {osm_to_fts[n]: (d["x"], d["y"]) for n, d in G_proj.nodes(data=True)}
    pos_latlon = {osm_to_fts[n]: (d["x"], d["y"]) for n, d in G_ll.nodes(data=True)}

    # 5. Process Edges
    best_per_uv = {} 
    for u, v, k, data in G_proj.edges(keys=True, data=True):
        length = float(data.get("length", 100.0))
        if (u, v) not in best_per_uv or length < best_per_uv[(u, v)][0]:
            best_per_uv[(u, v)] = (length, k, data)

    rows = []
    edge_geometries = {}
    
    # Absolute fallbacks if both the graph AND the user-provided defaults are None
    ABS_SPEED_KPH = 30.0
    ABS_LANES = 1

    for idx, ((u, v), (length, key, data)) in enumerate(best_per_uv.items()):
        # --- Speed Handling (km/h) ---
        raw_speed = data.get("speed_kph")
        if isinstance(raw_speed, list): raw_speed = raw_speed[0]
        
        try:
            speed = float(raw_speed)
            if np.isnan(speed): raise ValueError
        except (ValueError, TypeError):
            speed = float(default_speed) if default_speed is not None else ABS_SPEED_KPH

        # --- Lane Handling (Strict NaN Prevention) ---
        raw_lanes = data.get("lanes")
        # OSM lanes can be ['2', '3'] or "2" or None
        if isinstance(raw_lanes, list):
            raw_lanes = raw_lanes[0]
        
        try:
            # Handle potential strings like "2" or even "2.0"
            if raw_lanes is None:
                raise ValueError
            lanes = int(float(raw_lanes))
        except (ValueError, TypeError):
            # Fallback to user default, then to absolute floor
            lanes = default_lanes if default_lanes is not None else ABS_LANES
        
        # Final safety check: ensure lanes is a valid positive integer
        if lanes is None or (isinstance(lanes, float) and np.isnan(lanes)):
            lanes = ABS_LANES
        lanes = max(1, int(lanes))

        rows.append({
            "from": osm_to_fts[u],
            "to": osm_to_fts[v],
            "length": length,
            "speed": float(speed),
            "lanes": int(lanes),
        })

        # Geometry for rendering
        u_ll, v_ll = G_ll.nodes[u], G_ll.nodes[v]
        try:
            ll_data = G_ll.edges[u, v, key]
            coords = list(ll_data["geometry"].coords) if "geometry" in ll_data else [(u_ll["x"], u_ll["y"]), (v_ll["x"], v_ll["y"])]
        except:
            coords = [(u_ll["x"], u_ll["y"]), (v_ll["x"], v_ll["y"])]
        edge_geometries[idx] = coords

    return OSMnxResult(
        edges_df=pd.DataFrame(rows),
        pos_projected=pos_projected,
        pos_latlon=pos_latlon,
        edge_geometries=edge_geometries,
    )


# ---------------------------------------------------------------------------
# animate_map (Plotly Scattermap with OSM tiles)
# ---------------------------------------------------------------------------

def animate_map(
    edges_df,
    logs: np.ndarray,
    pos_latlon: Dict[Hashable, tuple],
    *,
    play_fps: int = 5,
    marker_size: int = 8,
    vehicle_colors: Optional[Dict[int, str]] = None,
    vehicle_ids: Optional[list[int]] = None,
    edge_geometries: Optional[Dict[int, list]] = None,
    edge_color: str = "rgba(50,50,50,0.5)",
    edge_width: float = 2.0,
    map_style: str = "open-street-map",
    map_zoom: Optional[int] = None,
    tween: bool = True,
) -> go.Figure:
    """Animate vehicles on a real map using Plotly Scattermap.

    Parameters
    ----------
    edges_df : DataFrame
        Edges table.
    logs : ndarray of shape ``(steps, n_vehicles, 2)``
        Position log (same format as :func:`animate`).
    pos_latlon : dict
        ``{node_id: (lon, lat)}`` positions in EPSG:4326.
    play_fps : int
        Playback frames per second.
    marker_size : int
        Vehicle marker diameter.
    vehicle_colors : dict or None
        ``{vehicle_id: color_string}``.
    vehicle_ids : list of int or None
        Subset of vehicle indices to draw.  If *None*, all vehicles are shown.
    edge_geometries : dict or None
        ``{edge_idx: [(lon, lat), ...]}`` from :func:`from_osmnx`.
        If *None*, straight lines between nodes are drawn.
    edge_color, edge_width
        Edge line styling.
    map_style : str
        Map tile style (``"open-street-map"``, ``"carto-positron"``, etc.).
    map_zoom : int or None
        Initial zoom level. Auto-computed if *None*.
    tween : bool
        Smooth interpolation between frames.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if logs.ndim != 3 or logs.shape[2] != 2:
        raise ValueError(f"logs must have shape (steps, n_vehicles, 2), got {logs.shape}")

    # --- vehicle subset filtering ---
    orig_ids = None
    if vehicle_ids is not None:
        orig_ids = np.asarray(sorted(vehicle_ids), dtype=int)
        logs = logs[:, orig_ids, :]
        if vehicle_colors is not None:
            vehicle_colors = {
                i: vehicle_colors[oid]
                for i, oid in enumerate(orig_ids)
                if oid in vehicle_colors
            }

    n_steps, n_veh, _ = logs.shape

    # --- map center & zoom ---
    lats = [pos_latlon[n][1] for n in pos_latlon]
    lons = [pos_latlon[n][0] for n in pos_latlon]
    center_lat, center_lon = float(np.mean(lats)), float(np.mean(lons))
    if map_zoom is None:
        max_range = max(max(lats) - min(lats), max(lons) - min(lons))
        map_zoom = int(np.clip(14 - np.log2(max_range / 0.01 + 1), 10, 18))

    # --- edge lines on map ---
    edge_lats: list = []
    edge_lons: list = []
    if edge_geometries:
        for idx in range(len(edges_df)):
            coords = edge_geometries.get(idx)
            if coords:
                for lon, lat in coords:
                    edge_lats.append(lat)
                    edge_lons.append(lon)
                edge_lats.append(None)
                edge_lons.append(None)
    else:
        from_col = edges_df["from"].values.astype(int)
        to_col = edges_df["to"].values.astype(int)
        for idx in range(len(edges_df)):
            u, v = int(from_col[idx]), int(to_col[idx])
            edge_lats.extend([pos_latlon[u][1], pos_latlon[v][1], None])
            edge_lons.extend([pos_latlon[u][0], pos_latlon[v][0], None])

    edge_trace = go.Scattermap(
        lat=edge_lats, lon=edge_lons, mode="lines",
        line=dict(width=edge_width, color=edge_color),
        hoverinfo="skip", showlegend=False,
    )

    fig = go.Figure(data=[edge_trace])

    if n_veh == 0:
        fig.update_layout(
            map=dict(style=map_style, center=dict(lat=center_lat, lon=center_lon), zoom=map_zoom),
        )
        fig.frames = []
        return fig

    # --- build per-edge lat/lon polyline geometry for interpolation ---
    map_geom = []
    from_col = edges_df["from"].values.astype(int)
    to_col = edges_df["to"].values.astype(int)
    length_col = edges_df["length"].values.astype(float)

    for idx in range(len(edges_df)):
        if edge_geometries and idx in edge_geometries:
            coords = edge_geometries[idx]
            lons_e = [c[0] for c in coords]
            lats_e = [c[1] for c in coords]
        else:
            u, v = int(from_col[idx]), int(to_col[idx])
            lons_e = [pos_latlon[u][0], pos_latlon[v][0]]
            lats_e = [pos_latlon[u][1], pos_latlon[v][1]]

        cum = _cumulative_arc(lons_e, lats_e)
        total = float(cum[-1]) if len(cum) > 0 else 0.0
        length_attr = float(length_col[idx]) if total > 0 else 1.0
        map_geom.append(dict(
            xs=np.asarray(lons_e), ys=np.asarray(lats_e),
            cum=cum, total=total, length_attr=length_attr,
            lanes=1, u=0, v=0,
        ))

    # --- precompute positions (reusing vectorized helper) ---
    # X will hold longitudes, Y will hold latitudes
    X, Y, OP = _precompute_positions(logs, map_geom)

    # --- colours ---
    colors, hover = _vehicle_colors_and_hover(logs, n_veh, vehicle_colors)
    if orig_ids is not None:
        hover = [h.replace(f"veh {i}", f"veh {orig_ids[i]}") for i, h in enumerate(hover)]
    hover_arr = np.array(hover, dtype=object)

    # --- initial vehicle trace ---
    text0 = np.where(OP[0] > 0, hover_arr, "")
    fig.add_trace(go.Scattermap(
        lat=Y[0], lon=X[0], mode="markers",
        marker=dict(size=marker_size, color=colors, opacity=OP[0]),
        hovertemplate="%{text}<extra></extra>", text=text0,
        showlegend=False,
    ))
    veh_idx = len(fig.data) - 1

    # --- frames ---
    frames = []
    for t in range(n_steps):
        text_t = np.where(OP[t] > 0, hover_arr, "")
        frames.append(go.Frame(
            name=str(t),
            data=[go.Scattermap(lat=Y[t], lon=X[t], marker=dict(opacity=OP[t]), text=text_t)],
            traces=[veh_idx],
        ))
    fig.frames = frames

    # --- layout ---
    fig.update_layout(
        map=dict(style=map_style, center=dict(lat=center_lat, lon=center_lon), zoom=map_zoom),
        margin=dict(l=0, r=0, t=0, b=90),
    )

    # --- controls ---
    ms = max(1, int(1000 / max(1, play_fps)))
    trans = {"duration": ms, "easing": "linear"} if tween else {"duration": 0}
    play_args = {"fromcurrent": True,
                 "frame": {"duration": ms, "redraw": True},
                 "transition": trans}

    fig.update_layout(
        sliders=[{
            "active": 0, "y": -0.02, "x": 0.15, "len": 0.72,
            "pad": {"t": 20, "b": 0},
            "currentvalue": {"prefix": "t = ", "visible": True},
            "steps": [
                {"args": [[str(t)], {"mode": "immediate",
                                     "frame": {"duration": 0, "redraw": True},
                                     "transition": {"duration": 0}}],
                 "label": str(t), "method": "animate"}
                for t in range(n_steps)
            ],
        }],
        updatemenus=[{
            "type": "buttons", "direction": "left",
            "x": 0.02, "y": -0.06, "showactive": True,
            "buttons": [{
                "label": "\u25b6 / \u23f8", "method": "animate",
                "args": [None, play_args],
                "args2": [[None], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": True},
                                   "transition": {"duration": 0}}],
            }],
        }],
    )

    return fig


# ---------------------------------------------------------------------------
# Occupancy helpers
# ---------------------------------------------------------------------------

_N_OCC_BINS = 10


def _compute_occupancy(logs: np.ndarray, n_edges: int) -> np.ndarray:
    """Compute per-edge vehicle count at each timestep.

    Returns array of shape ``(n_steps, n_edges)``.
    """
    n_steps, n_veh, _ = logs.shape
    occupancy = np.zeros((n_steps, n_edges), dtype=int)
    e_flat = logs[:, :, 0].astype(int)
    t_idx = np.repeat(np.arange(n_steps), n_veh)
    e_flat_1d = e_flat.ravel()
    mask = (e_flat_1d >= 0) & (e_flat_1d < n_edges)
    np.add.at(occupancy, (t_idx[mask], e_flat_1d[mask]), 1)
    return occupancy


def _occupancy_rgb(ratio: float) -> str:
    """Map ratio [0, 1] to green → yellow → red colour string."""
    r = max(0.0, min(1.0, ratio))
    if r <= 0.5:
        t = r * 2.0
        return f"rgb({int(255 * t)},{int(180 + 75 * t)},0)"
    else:
        t = (r - 0.5) * 2.0
        return f"rgb(255,{int(255 * (1 - t))},0)"


def _bin_colors() -> list[str]:
    """Return ``_N_OCC_BINS`` colours from green to red."""
    return [_occupancy_rgb(i / max(_N_OCC_BINS - 1, 1)) for i in range(_N_OCC_BINS)]


def _edge_segments_from_geom(geom: list[dict]) -> list[tuple[list, list]]:
    """Pre-extract per-edge (xs_with_None, ys_with_None) for fast concatenation."""
    segs = []
    for g in geom:
        xs = g["xs"].tolist() + [None]
        ys = g["ys"].tolist() + [None]
        segs.append((xs, ys))
    return segs


# ---------------------------------------------------------------------------
# animate_occupancy (Plotly, non-map)
# ---------------------------------------------------------------------------

def animate_occupancy(
    edges_df,
    logs: np.ndarray,
    pos: Optional[Dict[Hashable, tuple]] = None,
    *,
    capacity: Optional[np.ndarray | int | str | float] = None,
    play_fps: int = 5,
    edge_width: float = 5.0,
    tween: bool = True,
    max_frames: int = 200,
    edge_curvature: float = 0.22,
    base_offset: float = 0.55,
    parallel_spacing: float = 0.35,
    parallel_exponent: float = 1.6,
    traffic_rule: Literal["right", "left"] = "right",
    curve_single_edges: bool = False,
) -> go.Figure:
    """Animate edge occupancy with green → yellow → red colouring.

    Parameters
    ----------
    edges_df : DataFrame
        Edges table.
    logs : ndarray of shape ``(steps, n_vehicles, 2)``
        Position log.
    pos : dict or None
        Node positions.
    capacity : ndarray, int, float, or None
        Per-edge capacity (array of length ``n_edges``) or a single value
        applied to all edges.  If *None*, the maximum observed occupancy per
        edge is used as 100 %.
    play_fps : int
        Playback frames per second.
    edge_width : float
        Width of edge lines.
    tween : bool
        Smooth interpolation between frames.
    max_frames : int
        Maximum number of animation frames.  If the simulation has more
        timesteps, frames are uniformly subsampled.
    edge_curvature, base_offset, parallel_spacing, parallel_exponent,
    traffic_rule, curve_single_edges
        Edge geometry parameters (same as :func:`animate`).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if logs.ndim != 3 or logs.shape[2] != 2:
        raise ValueError(f"logs must have shape (steps, n_vehicles, 2), got {logs.shape}")

    n_steps_raw = logs.shape[0]
    n_edges = len(edges_df)
    if pos is None:
        pos = _auto_layout(edges_df)

    # --- occupancy ---
    occupancy = _compute_occupancy(logs, n_edges)

    # --- subsample if too many frames ---
    if n_steps_raw > max_frames:
        step_indices = np.linspace(0, n_steps_raw - 1, max_frames, dtype=int)
        occupancy = occupancy[step_indices]
    else:
        step_indices = np.arange(n_steps_raw)
    n_steps = len(step_indices)

    if capacity is None:
        cap = occupancy.max(axis=0).astype(float)
        cap = np.where(cap > 0, cap, 1.0)
    elif capacity == "DELTA":
        from fts import Simulator as _Simulator
        delta = float(_Simulator.DELTA)
        L = edges_df["length"].values.astype(float)
        lanes = edges_df["lanes"].values.astype(int)
        nmax_lane = np.floor(L / delta).astype(int) + 1
        nmax = np.maximum(1, lanes * nmax_lane)
        cap = nmax.astype(float)
    elif np.isscalar(capacity):
        cap = np.full(n_edges, float(capacity))
    else:
        cap = np.asarray(capacity, dtype=float)
        cap = np.where(cap > 0, cap, 1.0)

    # --- edge geometry ---
    geom = _build_edge_geometry(
        edges_df, pos,
        edge_curvature=edge_curvature, base_offset=base_offset,
        parallel_spacing=parallel_spacing, parallel_exponent=parallel_exponent,
        traffic_rule=traffic_rule, curve_single_edges=curve_single_edges,
        curve_samples=24,
    )
    segs = _edge_segments_from_geom(geom)

    # --- node trace ---
    nodes = sorted(set(edges_df["from"].astype(int)) | set(edges_df["to"].astype(int)))
    xN = [float(pos[n][0]) for n in nodes]
    yN = [float(pos[n][1]) for n in nodes]
    node_trace = go.Scatter(
        x=xN, y=yN, mode="markers",
        marker=dict(size=8, color="black"),
        hoverinfo="skip", showlegend=False,
    )

    fig = go.Figure(data=[node_trace])
    bin_colors = _bin_colors()

    # Helper: build bin traces for a given timestep
    def _bin_traces(t: int):
        ratios = np.clip(occupancy[t] / cap, 0.0, 1.0)
        bins = np.clip(
            (ratios * _N_OCC_BINS).astype(int), 0, _N_OCC_BINS - 1
        )
        # Edges with 0 occupancy get a light grey
        traces = []
        for b in range(_N_OCC_BINS):
            xs_all: list = []
            ys_all: list = []
            for ei in np.where(bins == b)[0]:
                xs_all.extend(segs[ei][0])
                ys_all.extend(segs[ei][1])
            # Also include unoccupied edges in bin 0
            traces.append(go.Scatter(
                x=xs_all or [None], y=ys_all or [None],
                mode="lines",
                line=dict(width=edge_width, color=bin_colors[b]),
                hoverinfo="skip", showlegend=False,
            ))
        return traces

    # Thin grey background edges (always visible, for context)
    bg_x: list = []
    bg_y: list = []
    for sx, sy in segs:
        bg_x.extend(sx)
        bg_y.extend(sy)
    fig.add_trace(go.Scatter(
        x=bg_x, y=bg_y, mode="lines",
        line=dict(width=max(1.0, edge_width * 0.4), color="rgba(200,200,200,0.5)"),
        hoverinfo="skip", showlegend=False,
    ))

    # --- initial bin traces ---
    initial_traces = _bin_traces(0)
    bin_start = len(fig.data)
    for tr in initial_traces:
        fig.add_trace(tr)
    bin_indices = list(range(bin_start, bin_start + _N_OCC_BINS))

    # --- hover trace at edge midpoints ---
    from_col = edges_df["from"].values.astype(int)
    to_col = edges_df["to"].values.astype(int)
    mid_x = [float(geom[i]["xs"][len(geom[i]["xs"]) // 2]) for i in range(n_edges)]
    mid_y = [float(geom[i]["ys"][len(geom[i]["ys"]) // 2]) for i in range(n_edges)]
    hover_text_0 = [
        f"edge {i} ({from_col[i]}→{to_col[i]})<br>occ: {occupancy[0, i]}"
        for i in range(n_edges)
    ]
    hover_trace = go.Scatter(
        x=mid_x, y=mid_y, mode="markers",
        marker=dict(size=12, opacity=0),
        hovertemplate="%{text}<extra></extra>", text=hover_text_0,
        showlegend=False,
    )
    fig.add_trace(hover_trace)
    hover_idx = len(fig.data) - 1

    # --- frames ---
    frames = []
    for t in range(n_steps):
        frame_traces = _bin_traces(t)
        hover_t = [
            f"edge {i} ({from_col[i]}→{to_col[i]})<br>occ: {occupancy[t, i]}"
            for i in range(n_edges)
        ]
        frame_traces.append(go.Scatter(text=hover_t))
        frames.append(go.Frame(
            name=str(t), data=frame_traces,
            traces=bin_indices + [hover_idx],
        ))
    fig.frames = frames

    # --- layout ---
    xs_all = [float(pos[n][0]) for n in pos]
    ys_all = [float(pos[n][1]) for n in pos]
    pad_x = (max(xs_all) - min(xs_all)) * 0.05 or 1.0
    pad_y = (max(ys_all) - min(ys_all)) * 0.05 or 1.0
    fig.update_xaxes(range=[min(xs_all) - pad_x, max(xs_all) + pad_x])
    fig.update_yaxes(range=[min(ys_all) - pad_y, max(ys_all) + pad_y],
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=90),
        xaxis=dict(showgrid=False, zeroline=False, visible=True),
        yaxis=dict(showgrid=False, zeroline=False, visible=True),
        plot_bgcolor="white",
    )

    # --- controls ---
    ms = max(1, int(1000 / max(1, play_fps)))
    trans = {"duration": ms, "easing": "linear"} if tween else {"duration": 0}
    play_args = {"fromcurrent": True, "frame": {"duration": ms, "redraw": True},
                 "transition": trans}
    slider_labels = [str(int(step_indices[t])) for t in range(n_steps)]
    fig.update_layout(
        sliders=[{
            "active": 0, "y": -0.07, "x": 0.15, "len": 0.72,
            "pad": {"t": 20, "b": 0},
            "currentvalue": {"prefix": "t = ", "visible": True},
            "steps": [
                {"args": [[str(t)], {"mode": "immediate",
                                     "frame": {"duration": 0, "redraw": True},
                                     "transition": {"duration": 0}}],
                 "label": slider_labels[t], "method": "animate"}
                for t in range(n_steps)
            ],
        }],
        updatemenus=[{
            "type": "buttons", "direction": "left",
            "x": 0.02, "y": -0.12, "showactive": True,
            "buttons": [{
                "label": "\u25b6 / \u23f8", "method": "animate",
                "args": [None, play_args],
                "args2": [[None], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": True},
                                   "transition": {"duration": 0}}],
            }],
        }],
    )

    return fig


# ---------------------------------------------------------------------------
# animate_occupancy_map (Plotly Scattermap with OSM tiles)
# ---------------------------------------------------------------------------

def animate_occupancy_map(
    edges_df,
    logs: np.ndarray,
    pos_latlon: Dict[Hashable, tuple],
    *,
    capacity: Optional[np.ndarray | int | str | float] = None,
    play_fps: int = 5,
    edge_width: float = 4.0,
    edge_geometries: Optional[Dict[int, list]] = None,
    map_style: str = "open-street-map",
    map_zoom: Optional[int] = None,
    tween: bool = True,
    max_frames: int = 200,
) -> go.Figure:
    """Animate edge occupancy on a real map with green → yellow → red colouring.

    Parameters
    ----------
    edges_df : DataFrame
        Edges table.
    logs : ndarray of shape ``(steps, n_vehicles, 2)``
        Position log.
    pos_latlon : dict
        ``{node_id: (lon, lat)}`` positions in EPSG:4326.
    capacity : ndarray, int, float, or None
        Per-edge capacity or scalar.  If *None*, uses max observed occupancy.
    play_fps : int
        Playback frames per second.
    edge_width : float
        Width of edge lines.
    edge_geometries : dict or None
        ``{edge_idx: [(lon, lat), ...]}`` from :func:`from_osmnx`.
    map_style : str
        Map tile style.
    map_zoom : int or None
        Initial zoom level.  Auto-computed if *None*.
    tween : bool
        Smooth interpolation between frames.
    max_frames : int
        Maximum number of animation frames.  If the simulation has more
        timesteps, frames are uniformly subsampled.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if logs.ndim != 3 or logs.shape[2] != 2:
        raise ValueError(f"logs must have shape (steps, n_vehicles, 2), got {logs.shape}")

    n_steps_raw = logs.shape[0]
    n_edges = len(edges_df)

    # --- map centre & zoom ---
    lats = [pos_latlon[n][1] for n in pos_latlon]
    lons = [pos_latlon[n][0] for n in pos_latlon]
    center_lat, center_lon = float(np.mean(lats)), float(np.mean(lons))
    if map_zoom is None:
        max_range = max(max(lats) - min(lats), max(lons) - min(lons))
        map_zoom = int(np.clip(14 - np.log2(max_range / 0.01 + 1), 10, 18))

    # --- occupancy ---
    occupancy = _compute_occupancy(logs, n_edges)

    # --- subsample if too many frames ---
    if n_steps_raw > max_frames:
        step_indices = np.linspace(0, n_steps_raw - 1, max_frames, dtype=int)
        occupancy = occupancy[step_indices]
    else:
        step_indices = np.arange(n_steps_raw)
    n_steps = len(step_indices)

    if capacity is None:
        cap = occupancy.max(axis=0).astype(float)
        cap = np.where(cap > 0, cap, 1.0)
    elif capacity == "DELTA":
        from fts import Simulator as _Simulator
        delta = float(_Simulator.DELTA)
        L = edges_df["length"].values.astype(float)
        lanes = edges_df["lanes"].values.astype(int)
        nmax_lane = np.floor(L / delta).astype(int) + 1
        nmax = np.maximum(1, lanes * nmax_lane)
        cap = nmax.astype(float)
    elif np.isscalar(capacity):
        cap = np.full(n_edges, float(capacity))
    else:
        cap = np.asarray(capacity, dtype=float)
        cap = np.where(cap > 0, cap, 1.0)

    # --- pre-compute per-edge lat/lon segments ---
    from_col = edges_df["from"].values.astype(int)
    to_col = edges_df["to"].values.astype(int)
    edge_segs_lat: list[list] = []
    edge_segs_lon: list[list] = []
    for idx in range(n_edges):
        if edge_geometries and idx in edge_geometries:
            coords = edge_geometries[idx]
            lons_e = [c[0] for c in coords] + [None]
            lats_e = [c[1] for c in coords] + [None]
        else:
            u, v = int(from_col[idx]), int(to_col[idx])
            lons_e = [pos_latlon[u][0], pos_latlon[v][0], None]
            lats_e = [pos_latlon[u][1], pos_latlon[v][1], None]
        edge_segs_lon.append(lons_e)
        edge_segs_lat.append(lats_e)

    bin_colors = _bin_colors()

    def _bin_traces_map(t: int):
        ratios = np.clip(occupancy[t] / cap, 0.0, 1.0)
        bins = np.clip(
            (ratios * _N_OCC_BINS).astype(int), 0, _N_OCC_BINS - 1
        )
        traces = []
        for b in range(_N_OCC_BINS):
            lat_all: list = []
            lon_all: list = []
            for ei in np.where(bins == b)[0]:
                lat_all.extend(edge_segs_lat[ei])
                lon_all.extend(edge_segs_lon[ei])
            traces.append(go.Scattermap(
                lat=lat_all or [None], lon=lon_all or [None],
                mode="lines",
                line=dict(width=edge_width, color=bin_colors[b]),
                hoverinfo="skip", showlegend=False,
            ))
        return traces

    # Thin grey background edges
    bg_lat: list = []
    bg_lon: list = []
    for slat, slon in zip(edge_segs_lat, edge_segs_lon):
        bg_lat.extend(slat)
        bg_lon.extend(slon)

    fig = go.Figure(data=[go.Scattermap(
        lat=bg_lat, lon=bg_lon, mode="lines",
        line=dict(width=max(1.0, edge_width * 0.4), color="rgba(200,200,200,0.5)"),
        hoverinfo="skip", showlegend=False,
    )])

    # --- initial bin traces ---
    initial_traces = _bin_traces_map(0)
    bin_start = len(fig.data)
    for tr in initial_traces:
        fig.add_trace(tr)
    bin_indices = list(range(bin_start, bin_start + _N_OCC_BINS))

    # --- frames ---
    frames = []
    for t in range(n_steps):
        frames.append(go.Frame(
            name=str(t),
            data=_bin_traces_map(t),
            traces=bin_indices,
        ))
    fig.frames = frames

    # --- layout ---
    fig.update_layout(
        map=dict(style=map_style, center=dict(lat=center_lat, lon=center_lon), zoom=map_zoom),
        margin=dict(l=0, r=0, t=0, b=90),
    )

    # --- controls ---
    ms = max(1, int(1000 / max(1, play_fps)))
    trans = {"duration": ms, "easing": "linear"} if tween else {"duration": 0}
    play_args = {"fromcurrent": True, "frame": {"duration": ms, "redraw": True},
                 "transition": trans}
    fig.update_layout(
        sliders=[{
            "active": 0, "y": -0.02, "x": 0.15, "len": 0.72,
            "pad": {"t": 20, "b": 0},
            "currentvalue": {"prefix": "t = ", "visible": True},
            "steps": [
                {"args": [[str(t)], {"mode": "immediate",
                                     "frame": {"duration": 0, "redraw": True},
                                     "transition": {"duration": 0}}],
                 "label": str(t), "method": "animate"}
                for t in range(n_steps)
            ],
        }],
        updatemenus=[{
            "type": "buttons", "direction": "left",
            "x": 0.02, "y": -0.06, "showactive": True,
            "buttons": [{
                "label": "\u25b6 / \u23f8", "method": "animate",
                "args": [None, play_args],
                "args2": [[None], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": True},
                                   "transition": {"duration": 0}}],
            }],
        }],
    )

    return fig


# ---------------------------------------------------------------------------
# Graph analysis plots
# ---------------------------------------------------------------------------

def plot_edge_occupancy(
    logs: np.ndarray,
    edges_df,
    *,
    edge_indices: Optional[list[int]] = None,
) -> go.Figure:
    """Plot number of vehicles on each edge over time.

    Parameters
    ----------
    logs : ndarray of shape ``(steps, n_vehicles, 2)``
        Position log (same format as :func:`animate`).
    edges_df : DataFrame
        Edges table.
    edge_indices : list of int or None
        Subset of edge indices to plot.  If *None*, all edges are shown.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n_steps, n_veh, _ = logs.shape
    n_edges = len(edges_df)
    if edge_indices is None:
        edge_indices = list(range(n_edges))

    # Vectorized occupancy counting
    occupancy = np.zeros((n_steps, n_edges), dtype=int)
    e_flat = logs[:, :, 0].astype(int)
    t_idx = np.repeat(np.arange(n_steps), n_veh)
    e_flat_1d = e_flat.ravel()
    mask = (e_flat_1d >= 0) & (e_flat_1d < n_edges)
    np.add.at(occupancy, (t_idx[mask], e_flat_1d[mask]), 1)

    fig = go.Figure()
    from_col = edges_df["from"].values.astype(int)
    to_col = edges_df["to"].values.astype(int)
    for ei in edge_indices:
        u, v = int(from_col[ei]), int(to_col[ei])
        fig.add_trace(go.Scatter(
            x=list(range(n_steps)), y=occupancy[:, ei],
            mode="lines", name=f"edge {ei} ({u}\u2192{v})",
        ))
    fig.update_layout(
        xaxis_title="Time step", yaxis_title="Vehicles on edge",
        template="simple_white",
    )
    return fig


__all__ = [
    "plot_network",
    "animate",
    "animate_mpl",
    "animate_map",
    "animate_occupancy",
    "animate_occupancy_map",
    "from_osmnx",
    "plot_edge_occupancy",
    "OSMnxResult",
]
