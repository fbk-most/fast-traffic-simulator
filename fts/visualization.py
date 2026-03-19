"""Traffic network visualization and animation utilities.

Provides interactive visualizations for FTS simulations:

- :func:`plot_network` — static network diagram with curved edges and hover info.
- :func:`animate` — Plotly animated vehicle movement (XY or map via *pos_latlon*).
- :func:`animate_mpl` — Matplotlib animated vehicle movement (faster, supports map tiles).
- :func:`from_osmnx` — import a real road network from OpenStreetMap via OSMnx.
- :func:`animate_occupancy` — animated edge occupancy heatmap (XY or map via *pos_latlon*).
- :func:`animate_occupancy_mpl` — Matplotlib occupancy animation (supports mp4/gif export).
- :func:`plot_edge_occupancy` — vehicles-per-edge over time.

All functions accept the edges DataFrame used by :class:`fts.Simulator` and a
``pos`` dict mapping node indices to ``(x, y)`` coordinates.  If no positions
are supplied, a spring layout is computed automatically.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import namedtuple
from dataclasses import dataclass, field
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
# Rendering context — thin abstraction over Scatter vs Scattermap
# ---------------------------------------------------------------------------

@dataclass
class _RenderCtx:
    """Encapsulates the map-vs-XY difference for Plotly rendering.

    In **map mode** traces are ``go.Scattermap(lat=Y, lon=X)``;
    in **XY mode** they are ``go.Scatter(x=X, y=Y)``.
    This class provides factory methods that pick the right one, so callers
    never branch on the mode themselves.
    """
    use_map: bool = False
    # Map-only fields (ignored in XY mode)
    map_style: str = "carto-positron"
    map_zoom: Optional[int] = None
    center_lat: float = 0.0
    center_lon: float = 0.0

    # --- trace factories ---

    def scatter(self, xs, ys, **kwargs) -> go.BaseTraceType:
        """Create a markers/text trace with the correct coord keys."""
        if self.use_map:
            return go.Scattermap(lat=ys, lon=xs, **kwargs)
        return go.Scatter(x=xs, y=ys, **kwargs)

    def line(self, xs, ys, **kwargs) -> go.BaseTraceType:
        """Create a line trace with the correct coord keys."""
        if self.use_map:
            return go.Scattermap(lat=ys, lon=xs, mode="lines", **kwargs)
        return go.Scatter(x=xs, y=ys, mode="lines", **kwargs)

    # --- layout helpers ---

    def base_layout(self, *, margin: Optional[dict] = None) -> dict:
        """Return kwargs for ``fig.update_layout()``."""
        if self.use_map:
            return dict(
                map=dict(
                    style=self.map_style,
                    center=dict(lat=self.center_lat, lon=self.center_lon),
                    zoom=self.map_zoom,
                ),
                margin=margin or dict(l=0, r=0, t=0, b=0),
            )
        return dict(
            margin=margin or dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, visible=True),
            yaxis=dict(showgrid=False, zeroline=False, visible=True,
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="white",
        )

    def apply_axis_range(self, fig: go.Figure, pos: Dict[Hashable, tuple]) -> None:
        """Set axis ranges (XY mode only — map mode auto-fits)."""
        if self.use_map:
            return
        xs_all = [float(pos[n][0]) for n in pos]
        ys_all = [float(pos[n][1]) for n in pos]
        pad_x = (max(xs_all) - min(xs_all)) * 0.05 or 1.0
        pad_y = (max(ys_all) - min(ys_all)) * 0.05 or 1.0
        fig.update_xaxes(range=[min(xs_all) - pad_x, max(xs_all) + pad_x])
        fig.update_yaxes(range=[min(ys_all) - pad_y, max(ys_all) + pad_y],
                         scaleanchor="x", scaleratio=1)


def _make_render_ctx(
    pos_latlon: Optional[Dict[Hashable, tuple]],
    *,
    map_style: str = "carto-positron",
    map_zoom: Optional[int] = None,
    map_center: Optional[tuple] = None,
) -> _RenderCtx:
    """Build a ``_RenderCtx`` from the user-facing parameters."""
    if pos_latlon is None:
        return _RenderCtx(use_map=False)

    lats = [pos_latlon[n][1] for n in pos_latlon]
    lons = [pos_latlon[n][0] for n in pos_latlon]
    if map_center is not None:
        center_lat, center_lon = float(map_center[1]), float(map_center[0])
    else:
        center_lat, center_lon = float(np.mean(lats)), float(np.mean(lons))
    if map_zoom is None:
        max_range = max(max(lats) - min(lats), max(lons) - min(lons))
        map_zoom = int(np.clip(14 - np.log2(max_range / 0.01 + 1), 10, 18))

    return _RenderCtx(
        use_map=True,
        map_style=map_style,
        map_zoom=map_zoom,
        center_lat=center_lat,
        center_lon=center_lon,
    )


# Lane offset for lat/lon mode: ~3.5 m lane width at mid-latitude
# 1° lat ≈ 111 km → 3.5 m ≈ 3.15e-5°
_MAP_LANE_OFFSET = 0.000032


def _resolve_pos(
    pos: Optional[Dict[Hashable, tuple]],
    pos_latlon: Optional[Dict[Hashable, tuple]],
    edges_df,
) -> Dict[Hashable, tuple]:
    """Return the coordinate dict to use for geometry, auto-computing if needed."""
    if pos_latlon is not None:
        return pos_latlon
    if pos is not None:
        return pos
    return _auto_layout(edges_df)


def _resolve_geom(
    edges_df,
    pos: Dict[Hashable, tuple],
    *,
    edge_geometries: Optional[Dict[int, list]] = None,
    edge_curvature: float = 0.22,
    base_offset: float = 0.55,
    parallel_spacing: float = 0.35,
    parallel_exponent: float = 1.6,
    traffic_rule: str = "right",
    curve_single_edges: bool = False,
    curve_samples: int = 48,
) -> list[dict]:
    """Unified geometry resolution.

    When *edge_geometries* is provided (typically from :func:`from_osmnx`),
    builds geometry dicts from those polylines.  Otherwise falls through to
    :func:`_build_edge_geometry`.

    Always populates ``lanes`` from *edges_df* so lane offsets work correctly.
    """
    lanes_col = edges_df["lanes"].values.astype(int) if "lanes" in edges_df.columns else None

    if edge_geometries is not None:
        length_col = edges_df["length"].values.astype(float)
        from_col = edges_df["from"].values.astype(int)
        to_col = edges_df["to"].values.astype(int)
        geom = []
        for idx in range(len(edges_df)):
            if idx in edge_geometries:
                coords = edge_geometries[idx]
                xs_e = [c[0] for c in coords]
                ys_e = [c[1] for c in coords]
            else:
                u, v = int(from_col[idx]), int(to_col[idx])
                xs_e = [pos[u][0], pos[v][0]]
                ys_e = [pos[u][1], pos[v][1]]
            cum = _cumulative_arc(xs_e, ys_e)
            total = float(cum[-1]) if len(cum) > 0 else 0.0
            length_attr = float(length_col[idx]) if total > 0 else 1.0
            lanes_val = max(1, int(lanes_col[idx])) if lanes_col is not None else 1
            geom.append(dict(
                xs=np.asarray(xs_e), ys=np.asarray(ys_e),
                cum=cum, total=total, length_attr=length_attr,
                lanes=lanes_val,
                u=int(from_col[idx]), v=int(to_col[idx]),
            ))
        return geom

    return _build_edge_geometry(
        edges_df, pos,
        edge_curvature=edge_curvature, base_offset=base_offset,
        parallel_spacing=parallel_spacing, parallel_exponent=parallel_exponent,
        traffic_rule=traffic_rule, curve_single_edges=curve_single_edges,
        curve_samples=curve_samples,
    )


def _animation_layout(
    fig: go.Figure,
    n_steps: int,
    t_offset: int,
    play_fps: int,
    tween: bool,
    *,
    use_map: bool = False,
    step_labels: Optional[list[str]] = None,
) -> None:
    """Apply slider + play/pause button to a Plotly animation figure."""
    ms = max(1, int(1000 / max(1, play_fps)))
    trans = {"duration": ms, "easing": "linear"} if tween else {"duration": 0}
    # Scattermap needs redraw=True; XY Scatter works with False
    redraw = use_map
    play_args = {
        "fromcurrent": True,
        "frame": {"duration": ms, "redraw": redraw},
        "transition": trans,
    }

    if step_labels is None:
        step_labels = [str(t_offset + t) for t in range(n_steps)]

    # Slider and button Y positions differ slightly for map vs XY to
    # avoid overlapping the map attribution.
    slider_y = -0.02 if use_map else -0.07
    button_y = -0.06 if use_map else -0.12

    fig.update_layout(
        margin=dict(b=90) if not use_map else dict(l=0, r=0, t=0, b=90),
        sliders=[{
            "active": 0, "y": slider_y, "x": 0.15, "len": 0.72,
            "pad": {"t": 20, "b": 0},
            "currentvalue": {"prefix": "t = ", "visible": True},
            "steps": [
                {"args": [[str(t)], {"mode": "immediate",
                                     "frame": {"duration": 0, "redraw": redraw},
                                     "transition": {"duration": 0}}],
                 "label": step_labels[t], "method": "animate"}
                for t in range(n_steps)
            ],
        }],
        updatemenus=[{
            "type": "buttons", "direction": "left",
            "x": 0.02, "y": button_y, "showactive": True,
            "buttons": [{
                "label": "\u25b6 / \u23f8", "method": "animate",
                "args": [None, play_args],
                "args2": [[None], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": redraw},
                                   "transition": {"duration": 0}}],
            }],
        }],
    )


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
    X_df = pd.DataFrame(X).ffill().bfill()
    Y_df = pd.DataFrame(Y).ffill().bfill()
    # Use centroid of known positions instead of (0,0) to avoid Null Island in map mode
    x_fill = float(X_df.stack().mean()) if not X_df.isna().all().all() else 0.0
    y_fill = float(Y_df.stack().mean()) if not Y_df.isna().all().all() else 0.0
    X = X_df.fillna(x_fill).values
    Y = Y_df.fillna(y_fill).values

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
        f"veh {i} | OD {o}\u2192{d}" if o >= 0 else f"veh {i}"
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
    pos_latlon: Optional[Dict[Hashable, tuple]] = None,
    edge_geometries: Optional[Dict[int, list]] = None,
    map_style: str = "carto-positron",
    map_zoom: Optional[int] = None,
    map_center: Optional[tuple] = None,
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

    Supports both XY-plane mode and real-map mode.  If *pos_latlon* is
    provided the network is rendered on OpenStreetMap tiles using Plotly
    Scattermap; otherwise an XY Scatter plot is used.

    Parameters
    ----------
    edges_df : DataFrame
        Edges table with at least ``from``, ``to``, ``length``, ``speed``,
        ``lanes`` columns (same format accepted by :class:`fts.Simulator`).
    pos : dict or None
        Mapping ``{node_id: (x, y)}``.  If *None*, a spring layout is used.
        Ignored when *pos_latlon* is given.
    pos_latlon : dict or None
        Mapping ``{node_id: (lon, lat)}``.  When provided, the network is
        drawn on map tiles.
    edge_geometries : dict or None
        Mapping ``{edge_index: [(x, y), ...]}``.  Detailed polyline
        coordinates for edges (typically from :func:`from_osmnx`).
    map_style : str
        Plotly map style (e.g. ``"open-street-map"``, ``"carto-positron"``).
    map_zoom : int or None
        Map zoom level.  Auto-computed if *None*.
    map_center : tuple or None
        ``(lon, lat)`` centre of the map.  Auto-computed if *None*.
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
    ctx = _make_render_ctx(pos_latlon, map_style=map_style,
                           map_zoom=map_zoom, map_center=map_center)
    coords = _resolve_pos(pos, pos_latlon, edges_df)

    geom = _resolve_geom(
        edges_df, coords,
        edge_geometries=edge_geometries,
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

    edge_trace = ctx.line(
        all_ex, all_ey,
        line=dict(width=edge_width, color=edge_color),
        hoverinfo="skip", showlegend=False,
    )

    edge_hover = ctx.scatter(
        mid_x, mid_y, mode="markers",
        marker=dict(size=10, opacity=0),
        hovertemplate="%{text}<extra></extra>", text=mid_text,
        showlegend=False,
    )

    # --- node trace ---
    nodes = sorted(set(edges_df["from"].astype(int)) | set(edges_df["to"].astype(int)))
    xN = [float(coords[n][0]) for n in nodes]
    yN = [float(coords[n][1]) for n in nodes]
    node_text = [f"<b>node {n}</b>" for n in nodes]

    node_trace = ctx.scatter(
        xN, yN, mode="markers",
        marker=dict(size=node_size, color="black"),
        hovertemplate="%{text}<extra></extra>", text=node_text,
        name="nodes",
    )

    fig = go.Figure(data=[edge_trace, edge_hover, node_trace])
    fig.update_layout(**ctx.base_layout())
    return fig


# ---------------------------------------------------------------------------
# animate (Plotly)
# ---------------------------------------------------------------------------

def animate(
    edges_df,
    logs: np.ndarray,
    pos: Optional[Dict[Hashable, tuple]] = None,
    *,
    pos_latlon: Optional[Dict[Hashable, tuple]] = None,
    edge_geometries: Optional[Dict[int, list]] = None,
    map_style: str = "carto-positron",
    map_zoom: Optional[int] = None,
    map_center: Optional[tuple] = None,
    play_fps: int = 5,
    marker_size: int = 8,
    lane_offset: float = 0.06,
    tween: bool = True,
    lane_logs: Optional[np.ndarray] = None,
    vehicle_colors: Optional[Dict[int, str]] = None,
    vehicle_ids: Optional[list[int]] = None,
    edge_color: str = "rgba(50,50,50,0.5)",
    edge_width: float = 2.0,
    edge_curvature: float = 0.22,
    base_offset: float = 0.55,
    parallel_spacing: float = 0.35,
    parallel_exponent: float = 1.6,
    traffic_rule: Literal["right", "left"] = "right",
    curve_single_edges: bool = False,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
) -> go.Figure:
    """Animate vehicle movement on the network.

    Supports both XY-plane mode and real-map mode.  If *pos_latlon* is
    provided the animation is rendered on OpenStreetMap tiles using Plotly
    Scattermap; otherwise an XY Scatter plot is used.

    Parameters
    ----------
    edges_df : DataFrame
        Edges table (same as :func:`plot_network`).
    logs : ndarray of shape ``(steps, n_vehicles, 2)``
        Position log.  For each vehicle at each step:
        ``[edge_index, edge_distance]``.  Edge index ``-1`` means the
        vehicle is not currently on an edge (waiting/arrived).
    pos : dict or None
        Node positions ``{node_id: (x, y)}`` for XY mode.
        Auto-computed if *None* and *pos_latlon* is not given.
    pos_latlon : dict or None
        ``{node_id: (lon, lat)}`` in EPSG:4326 for map mode.
    edge_geometries : dict or None
        ``{edge_idx: [(x, y), ...]}`` from :func:`from_osmnx`.
    map_style : str
        Map tile style for map mode.
    map_zoom : int or None
        Initial zoom level for map mode.  Auto-computed if *None*.
    map_center : tuple or None
        ``(lon, lat)`` for the initial map centre.  Defaults to the
        centroid of all nodes.
    play_fps : int
        Playback frames per second.
    marker_size : int
        Vehicle marker diameter.
    lane_offset : float
        Lateral offset per lane (XY mode).  In map mode an appropriate
        offset in degrees is used automatically.
    tween : bool
        Smooth interpolation between frames.
    t_start : int or None
        First timestep to include (default: 0).
    t_end : int or None
        One past the last timestep to include (default: all steps).
    lane_logs : ndarray or None
        Per-step lane index for each vehicle.
    vehicle_colors : dict or None
        ``{vehicle_id: color_string}``.
    vehicle_ids : list of int or None
        Subset of vehicle indices to draw.
    edge_color : str
        Edge line colour.
    edge_width : float
        Edge line width.
    edge_curvature, base_offset, parallel_spacing, parallel_exponent,
    traffic_rule, curve_single_edges
        Edge geometry parameters.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ctx = _make_render_ctx(pos_latlon, map_style=map_style,
                           map_zoom=map_zoom, map_center=map_center)
    coords = _resolve_pos(pos, pos_latlon, edges_df)

    if logs.ndim != 3 or logs.shape[2] != 2:
        raise ValueError(
            f"logs must have shape (steps, n_vehicles, 2), got {logs.shape}"
        )

    # --- time range slicing ---
    _t0 = t_start if t_start is not None else 0
    _t1 = t_end if t_end is not None else logs.shape[0]
    t_offset = _t0
    logs = logs[_t0:_t1]
    if lane_logs is not None:
        lane_logs = lane_logs[_t0:_t1]

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

    # --- base figure with edge lines ---
    base_fig = plot_network(
        edges_df, pos, pos_latlon=pos_latlon,
        edge_geometries=edge_geometries,
        map_style=map_style, map_zoom=map_zoom, map_center=map_center,
        edge_color=edge_color, edge_width=edge_width,
        edge_curvature=edge_curvature, base_offset=base_offset,
        parallel_spacing=parallel_spacing, parallel_exponent=parallel_exponent,
        traffic_rule=traffic_rule, curve_single_edges=curve_single_edges,
    )
    fig = go.Figure(data=list(base_fig.data), layout=base_fig.layout)

    if n_veh == 0:
        fig.frames = []
        return fig

    # --- geometry + position precomputation ---
    geom = _resolve_geom(
        edges_df, coords,
        edge_geometries=edge_geometries,
        edge_curvature=edge_curvature, base_offset=base_offset,
        parallel_spacing=parallel_spacing, parallel_exponent=parallel_exponent,
        traffic_rule=traffic_rule, curve_single_edges=curve_single_edges,
        curve_samples=48,
    )

    effective_lane_offset = _MAP_LANE_OFFSET if ctx.use_map else lane_offset
    X, Y, OP = _precompute_positions(
        logs, geom,
        lane_logs=lane_logs, lane_offset=effective_lane_offset,
        traffic_rule=traffic_rule,
    )

    colors, hover = _vehicle_colors_and_hover(logs, n_veh, vehicle_colors)
    if orig_ids is not None:
        hover = [h.replace(f"veh {i}", f"veh {orig_ids[i]}") for i, h in enumerate(hover)]
    hover_arr = np.array(hover, dtype=object)

    # --- vehicle trace + frames ---
    marker_kw: dict = dict(size=marker_size, color=colors, opacity=OP[0])
    if not ctx.use_map:
        marker_kw.update(symbol="circle", line=dict(color="black", width=1))

    text0 = np.where(OP[0] > 0, hover_arr, "")
    fig.add_trace(ctx.scatter(
        X[0], Y[0], mode="markers",
        marker=marker_kw,
        hovertemplate="%{text}<extra></extra>", text=text0,
        showlegend=False,
    ))
    veh_idx = len(fig.data) - 1

    frames = []
    for t in range(n_steps):
        text_t = np.where(OP[t] > 0, hover_arr, "")
        frames.append(go.Frame(
            name=str(t),
            data=[ctx.scatter(
                X[t], Y[t],
                marker=dict(opacity=OP[t]), text=text_t,
            )],
            traces=[veh_idx],
        ))
    fig.frames = frames

    # --- layout + animation controls ---
    ctx.apply_axis_range(fig, coords)
    _animation_layout(fig, n_steps, t_offset, play_fps, tween,
                       use_map=ctx.use_map)

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
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
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

    # --- time range slicing ---
    _t0 = t_start if t_start is not None else 0
    _t1 = t_end if t_end is not None else logs.shape[0]
    t_offset = _t0
    logs = logs[_t0:_t1]
    if lane_logs is not None:
        lane_logs = lane_logs[_t0:_t1]

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

    title_text = ax.set_title(f"t = {t_offset}")

    def _init():
        return (scat, title_text)

    def _update(frame):
        offsets = np.column_stack([X[frame], Y[frame]])
        scat.set_offsets(offsets)
        scat.set_alpha(OP[frame])
        title_text.set_text(f"t = {t_offset + frame}")
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
    "edge_geometries_projected", "crs",
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
    """Import a road network from OSM via OSMnx with strict NaN prevention.
    
    bbox format: (west, south, east, north)
    """
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
    edge_geometries_proj = {}

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

        # Geometry for rendering (lon/lat for Plotly map mode)
        u_ll, v_ll = G_ll.nodes[u], G_ll.nodes[v]
        try:
            ll_data = G_ll.edges[u, v, key]
            coords = list(ll_data["geometry"].coords) if "geometry" in ll_data else [(u_ll["x"], u_ll["y"]), (v_ll["x"], v_ll["y"])]
        except:
            coords = [(u_ll["x"], u_ll["y"]), (v_ll["x"], v_ll["y"])]
        edge_geometries[idx] = coords

        # Projected geometry for Matplotlib map mode
        u_pr, v_pr = G_proj.nodes[u], G_proj.nodes[v]
        try:
            pr_data = G_proj.edges[u, v, key]
            coords_proj = list(pr_data["geometry"].coords) if "geometry" in pr_data else [(u_pr["x"], u_pr["y"]), (v_pr["x"], v_pr["y"])]
        except:
            coords_proj = [(u_pr["x"], u_pr["y"]), (v_pr["x"], v_pr["y"])]
        edge_geometries_proj[idx] = coords_proj

    proj_crs = str(G_proj.graph.get("crs", ""))

    return OSMnxResult(
        edges_df=pd.DataFrame(rows),
        pos_projected=pos_projected,
        pos_latlon=pos_latlon,
        edge_geometries=edge_geometries,
        edge_geometries_projected=edge_geometries_proj,
        crs=proj_crs,
    )


# ---------------------------------------------------------------------------
# animate_map (deprecated — use animate(pos_latlon=...))
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
    map_style: str = "carto-positron",
    map_zoom: Optional[int] = None,
    tween: bool = True,
) -> go.Figure:
    """Deprecated — use ``animate(pos_latlon=...)`` instead."""
    import warnings
    warnings.warn(
        "animate_map() is deprecated, use animate(pos_latlon=...) instead.",
        DeprecationWarning, stacklevel=2,
    )
    return animate(
        edges_df, logs, pos_latlon=pos_latlon,
        play_fps=play_fps, marker_size=marker_size,
        vehicle_colors=vehicle_colors, vehicle_ids=vehicle_ids,
        edge_geometries=edge_geometries, edge_color=edge_color,
        edge_width=edge_width, map_style=map_style, map_zoom=map_zoom,
        tween=tween,
    )


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


def compute_edge_capacity(edges_df, delta: float) -> np.ndarray:
    """Compute per-edge vehicle capacity from edge geometry.

    Parameters
    ----------
    edges_df : DataFrame
        Edges table with ``length`` and ``lanes`` columns.
    delta : float
        Minimum safe following distance (metres).

    Returns
    -------
    ndarray of shape ``(n_edges,)``
        Maximum number of vehicles that fit on each edge.
    """
    L = edges_df["length"].values.astype(float)
    lanes = edges_df["lanes"].values.astype(int)
    nmax_lane = np.floor(L / delta).astype(int) + 1
    return np.maximum(1, lanes * nmax_lane)


def compute_occupancy(logs: np.ndarray, edges_df, delta: float) -> np.ndarray:
    """Compute per-edge occupancy ratio at each timestep.

    Parameters
    ----------
    logs : ndarray of shape ``(steps, n_vehicles, 2)``
        Position log (edge index + distance) as produced by a simulation loop.
    edges_df : DataFrame
        Edges table with ``length`` and ``lanes`` columns.
    delta : float
        Minimum safe following distance (metres) used in the simulation.

    Returns
    -------
    ndarray of shape ``(n_steps, n_edges)``
        Occupancy ratio in [0, 1] for each edge at each timestep.
    """
    n_edges = len(edges_df)
    counts = _compute_occupancy(logs, n_edges)
    cap = compute_edge_capacity(edges_df, delta).astype(float)
    return np.clip(counts / cap[np.newaxis, :], 0.0, 1.0)


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
# animate_occupancy (Plotly, unified: XY + map)
# ---------------------------------------------------------------------------

def animate_occupancy(
    edges_df,
    occupancy: np.ndarray,
    pos: Optional[Dict[Hashable, tuple]] = None,
    *,
    pos_latlon: Optional[Dict[Hashable, tuple]] = None,
    edge_geometries: Optional[Dict[int, list]] = None,
    map_style: str = "carto-positron",
    map_zoom: Optional[int] = None,
    map_center: Optional[tuple] = None,
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
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
) -> go.Figure:
    """Animate edge occupancy with green -> yellow -> red colouring.

    Supports both XY-plane mode and real-map mode.  If *pos_latlon* is
    provided the animation is rendered on OpenStreetMap tiles; otherwise
    an XY Scatter plot is used.

    Parameters
    ----------
    edges_df : DataFrame
        Edges table.
    occupancy : ndarray of shape ``(n_steps, n_edges)``
        Pre-computed occupancy ratios in [0, 1] (e.g. from
        :func:`compute_occupancy`).
    pos : dict or None
        Node positions for XY mode.  Auto-computed if *None*.
    pos_latlon : dict or None
        ``{node_id: (lon, lat)}`` in EPSG:4326 for map mode.
    edge_geometries : dict or None
        ``{edge_idx: [(x, y), ...]}`` from :func:`from_osmnx`.
    map_style : str
        Map tile style.
    map_zoom : int or None
        Initial zoom level.  Auto-computed if *None*.
    map_center : tuple or None
        ``(lon, lat)`` for the initial map centre.  Defaults to the
        centroid of all nodes.
    play_fps : int
        Playback frames per second.
    edge_width : float
        Width of edge lines.
    tween : bool
        Smooth interpolation between frames.
    max_frames : int
        Maximum animation frames (subsampled if exceeded).
    edge_curvature, base_offset, parallel_spacing, parallel_exponent,
    traffic_rule, curve_single_edges
        Edge geometry parameters.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ctx = _make_render_ctx(pos_latlon, map_style=map_style,
                           map_zoom=map_zoom, map_center=map_center)
    coords = _resolve_pos(pos, pos_latlon, edges_df)

    if occupancy.ndim != 2:
        raise ValueError(f"occupancy must have shape (n_steps, n_edges), got {occupancy.shape}")

    # --- time range slicing ---
    _t0 = t_start if t_start is not None else 0
    _t1 = t_end if t_end is not None else occupancy.shape[0]
    t_offset = _t0
    occupancy = occupancy[_t0:_t1]

    n_steps_raw = occupancy.shape[0]
    n_edges = len(edges_df)

    # --- subsample ---
    if n_steps_raw > max_frames:
        step_indices = np.linspace(0, n_steps_raw - 1, max_frames, dtype=int)
        occupancy = occupancy[step_indices]
    else:
        step_indices = np.arange(n_steps_raw)
    n_steps = len(step_indices)

    from_col = edges_df["from"].values.astype(int)
    to_col = edges_df["to"].values.astype(int)

    # --- per-edge colour at each timestep (precompute) ---
    ratios_all = np.clip(occupancy, 0.0, 1.0)

    def _edge_color(ratio: float) -> str:
        return _occupancy_rgb(ratio)

    # --- edge geometry ---
    geom = _resolve_geom(
        edges_df, coords,
        edge_geometries=edge_geometries,
        edge_curvature=edge_curvature, base_offset=base_offset,
        parallel_spacing=parallel_spacing, parallel_exponent=parallel_exponent,
        traffic_rule=traffic_rule, curve_single_edges=curve_single_edges,
        curve_samples=24,
    )
    segs = _edge_segments_from_geom(geom)

    # --- base figure: nodes + background edges ---
    if not ctx.use_map:
        nodes = sorted(set(edges_df["from"].astype(int)) | set(edges_df["to"].astype(int)))
        xN = [float(coords[n][0]) for n in nodes]
        yN = [float(coords[n][1]) for n in nodes]
        node_trace = ctx.scatter(
            xN, yN, mode="markers",
            marker=dict(size=8, color="black"),
            hoverinfo="skip", showlegend=False,
        )
        fig = go.Figure(data=[node_trace])
    else:
        fig = go.Figure()

    bg_x: list = []
    bg_y: list = []
    for sx, sy in segs:
        bg_x.extend(sx)
        bg_y.extend(sy)
    fig.add_trace(ctx.line(
        bg_x, bg_y,
        line=dict(width=max(1.0, edge_width * 0.4), color="rgba(200,200,200,0.5)"),
        hoverinfo="skip", showlegend=False,
    ))

    # Use per-edge traces: each edge gets a fixed-geometry trace.
    # Frames only update line.color — no coordinate interpolation, no jumpiness.
    # For very large networks (>500 edges), fall back to bin grouping with tween off.
    use_per_edge = n_edges <= 500

    if use_per_edge:
        edge_start = len(fig.data)
        for ei in range(n_edges):
            xs_e = geom[ei]["xs"].tolist()
            ys_e = geom[ei]["ys"].tolist()
            fig.add_trace(ctx.line(
                xs_e, ys_e,
                line=dict(width=edge_width, color=_edge_color(ratios_all[0, ei])),
                hoverinfo="skip", showlegend=False,
            ))
        edge_trace_indices = list(range(edge_start, edge_start + n_edges))

        # Hover trace at edge midpoints
        mid_x = [float(geom[i]["xs"][len(geom[i]["xs"]) // 2]) for i in range(n_edges)]
        mid_y = [float(geom[i]["ys"][len(geom[i]["ys"]) // 2]) for i in range(n_edges)]
        hover_text_0 = [
            f"edge {i} ({from_col[i]}\u2192{to_col[i]})<br>occ: {occupancy[0, i]:.0%}"
            for i in range(n_edges)
        ]
        fig.add_trace(ctx.scatter(
            mid_x, mid_y, mode="markers",
            marker=dict(size=12, opacity=0),
            hovertemplate="%{text}<extra></extra>", text=hover_text_0,
            showlegend=False,
        ))
        hover_idx = len(fig.data) - 1

        frames = []
        for t in range(n_steps):
            frame_data = []
            for ei in range(n_edges):
                xs_e = geom[ei]["xs"].tolist()
                ys_e = geom[ei]["ys"].tolist()
                frame_data.append(ctx.line(
                    xs_e, ys_e,
                    line=dict(width=edge_width, color=_edge_color(ratios_all[t, ei])),
                    hoverinfo="skip", showlegend=False,
                ))
            hover_t = [
                f"edge {i} ({from_col[i]}\u2192{to_col[i]})<br>occ: {occupancy[t, i]:.0%}"
                for i in range(n_edges)
            ]
            frame_data.append(ctx.scatter([], [], text=hover_t))
            frames.append(go.Frame(
                name=str(t), data=frame_data,
                traces=edge_trace_indices + [hover_idx],
            ))
        fig.frames = frames
    else:
        # Bin grouping fallback for large networks — force tween off
        tween = False
        bin_colors = _bin_colors()

        def _bin_traces(t: int):
            bins = np.clip(
                (ratios_all[t] * _N_OCC_BINS).astype(int), 0, _N_OCC_BINS - 1
            )
            traces = []
            for b in range(_N_OCC_BINS):
                xs_all: list = []
                ys_all: list = []
                for ei in np.where(bins == b)[0]:
                    xs_all.extend(segs[ei][0])
                    ys_all.extend(segs[ei][1])
                traces.append(ctx.line(
                    xs_all or [None], ys_all or [None],
                    line=dict(width=edge_width, color=bin_colors[b]),
                    hoverinfo="skip", showlegend=False,
                ))
            return traces

        initial_traces = _bin_traces(0)
        bin_start = len(fig.data)
        for tr in initial_traces:
            fig.add_trace(tr)
        bin_indices_list = list(range(bin_start, bin_start + _N_OCC_BINS))

        mid_x = [float(geom[i]["xs"][len(geom[i]["xs"]) // 2]) for i in range(n_edges)]
        mid_y = [float(geom[i]["ys"][len(geom[i]["ys"]) // 2]) for i in range(n_edges)]
        hover_text_0 = [
            f"edge {i} ({from_col[i]}\u2192{to_col[i]})<br>occ: {occupancy[0, i]:.0%}"
            for i in range(n_edges)
        ]
        fig.add_trace(ctx.scatter(
            mid_x, mid_y, mode="markers",
            marker=dict(size=12, opacity=0),
            hovertemplate="%{text}<extra></extra>", text=hover_text_0,
            showlegend=False,
        ))
        hover_idx = len(fig.data) - 1

        frames = []
        for t in range(n_steps):
            frame_traces = _bin_traces(t)
            hover_t = [
                f"edge {i} ({from_col[i]}\u2192{to_col[i]})<br>occ: {occupancy[t, i]:.0%}"
                for i in range(n_edges)
            ]
            frame_traces.append(ctx.scatter([], [], text=hover_t))
            frames.append(go.Frame(
                name=str(t), data=frame_traces,
                traces=bin_indices_list + [hover_idx],
            ))
        fig.frames = frames

    # --- layout + animation controls ---
    fig.update_layout(**ctx.base_layout())
    ctx.apply_axis_range(fig, coords)
    slider_labels = [str(t_offset + int(step_indices[t])) for t in range(n_steps)]
    _animation_layout(fig, n_steps, t_offset, play_fps, tween,
                       use_map=ctx.use_map, step_labels=slider_labels)

    return fig


# ---------------------------------------------------------------------------
# animate_occupancy_map (deprecated — use animate_occupancy(pos_latlon=...))
# ---------------------------------------------------------------------------

def animate_occupancy_map(
    edges_df,
    occupancy: np.ndarray,
    pos_latlon: Dict[Hashable, tuple],
    *,
    play_fps: int = 5,
    edge_width: float = 4.0,
    edge_geometries: Optional[Dict[int, list]] = None,
    map_style: str = "carto-positron",
    map_zoom: Optional[int] = None,
    tween: bool = True,
    max_frames: int = 200,
) -> go.Figure:
    """Deprecated — use ``animate_occupancy(pos_latlon=...)`` instead."""
    import warnings
    warnings.warn(
        "animate_occupancy_map() is deprecated, use animate_occupancy(pos_latlon=...) instead.",
        DeprecationWarning, stacklevel=2,
    )
    return animate_occupancy(
        edges_df, occupancy, pos_latlon=pos_latlon,
        play_fps=play_fps, edge_width=edge_width,
        edge_geometries=edge_geometries, map_style=map_style,
        map_zoom=map_zoom, tween=tween, max_frames=max_frames,
    )


# ---------------------------------------------------------------------------
# animate_occupancy_mpl (Matplotlib + optional map tiles)
# ---------------------------------------------------------------------------

def animate_occupancy_mpl(
    edges_df,
    occupancy: np.ndarray,
    pos: Optional[Dict[Hashable, tuple]] = None,
    *,
    play_fps: int = 10,
    edge_width: float = 3.0,
    max_frames: int = 200,
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
    edge_geometries: Optional[Dict[int, list]] = None,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
):
    """Animate edge occupancy using Matplotlib (supports mp4/gif export).

    Uses a ``LineCollection`` whose per-segment colours update each frame —
    geometry stays fixed, so there is no jumpiness.

    Parameters
    ----------
    edges_df : DataFrame
        Edges table.
    occupancy : ndarray of shape ``(n_steps, n_edges)``
        Pre-computed occupancy ratios in [0, 1] (e.g. from
        :func:`compute_occupancy`).
    pos : dict or None
        Node positions.  Auto-computed if *None* and *edge_geometries* is
        not provided.
    play_fps : int
        Playback frames per second.
    edge_width : float
        Width of edge lines.
    max_frames : int
        Maximum animation frames.
    edge_curvature, base_offset, parallel_spacing, parallel_exponent,
    traffic_rule, curve_single_edges
        Edge geometry parameters (ignored when *edge_geometries* is given).
    save_path : str or None
        Save to file (e.g. ``"out.mp4"`` or ``"out.gif"``).
    dpi : int
        Resolution for saved animation.
    figsize : tuple
        Figure size in inches.
    tile_source
        Contextily tile provider.  Defaults to OpenStreetMap if *crs* is set.
    crs : str or None
        CRS of *pos* (e.g. ``"EPSG:3857"``).  Enables map tiles.
    edge_geometries : dict or None
        ``{edge_idx: [(x, y), ...]}`` polylines per edge.  When provided,
        these are used instead of ``_build_edge_geometry()``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import LineCollection
    except ImportError:
        raise ImportError("matplotlib is required for animate_occupancy_mpl(). Install with: pip install matplotlib")

    if occupancy.ndim != 2:
        raise ValueError(f"occupancy must have shape (n_steps, n_edges), got {occupancy.shape}")

    # --- time range slicing ---
    _t0 = t_start if t_start is not None else 0
    _t1 = t_end if t_end is not None else occupancy.shape[0]
    t_offset = _t0
    occupancy = occupancy[_t0:_t1]

    n_steps_raw = occupancy.shape[0]
    n_edges = len(edges_df)

    # --- subsample ---
    if n_steps_raw > max_frames:
        step_indices = np.linspace(0, n_steps_raw - 1, max_frames, dtype=int)
        occupancy = occupancy[step_indices]
    else:
        step_indices = np.arange(n_steps_raw)
    n_steps = len(step_indices)

    # --- build segments ---
    if edge_geometries is not None:
        segments = []
        for idx in range(n_edges):
            if idx in edge_geometries:
                coords = edge_geometries[idx]
                pts = np.array(coords, dtype=float)
            else:
                # Fallback to straight line
                from_col = edges_df["from"].values.astype(int)
                to_col = edges_df["to"].values.astype(int)
                u, v = int(from_col[idx]), int(to_col[idx])
                pts = np.array([list(pos[u]), list(pos[v])], dtype=float) if pos else np.array([[0, 0], [1, 1]])
            segments.append(pts)
    else:
        if pos is None:
            pos = _auto_layout(edges_df)
        geom = _build_edge_geometry(
            edges_df, pos,
            edge_curvature=edge_curvature, base_offset=base_offset,
            parallel_spacing=parallel_spacing, parallel_exponent=parallel_exponent,
            traffic_rule=traffic_rule, curve_single_edges=curve_single_edges,
            curve_samples=24,
        )
        segments = [np.column_stack([g["xs"], g["ys"]]) for g in geom]

    # --- green -> yellow -> red colourmap ---
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "occ", [(0, 0.71, 0), (1, 1, 0), (1, 0, 0)], N=256,
    )

    # --- figure ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    lc = LineCollection(segments, linewidths=edge_width, cmap=cmap, clim=(0, 1))
    ratios_0 = np.clip(occupancy[0], 0.0, 1.0)
    lc.set_array(ratios_0)
    ax.add_collection(lc)

    # Nodes
    if pos is not None:
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
            pass

    # Auto-range
    all_pts = np.vstack(segments)
    x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
    pad_x = (x_max - x_min) * 0.05 or 1.0
    pad_y = (y_max - y_min) * 0.05 or 1.0
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    fig.colorbar(lc, ax=ax, label="Occupancy ratio", shrink=0.6)
    title_text = ax.set_title(f"t = {t_offset}")

    def _init():
        return (lc, title_text)

    def _update(frame):
        ratios = np.clip(occupancy[frame], 0.0, 1.0)
        lc.set_array(ratios)
        title_text.set_text(f"t = {t_offset + int(step_indices[frame])}")
        return (lc, title_text)

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
# Graph analysis plots
# ---------------------------------------------------------------------------

def plot_edge_occupancy(
    occupancy: np.ndarray,
    edges_df,
    *,
    edge_indices: Optional[list[int]] = None,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
) -> go.Figure:
    """Plot per-edge occupancy ratio over time.

    Parameters
    ----------
    occupancy : ndarray of shape ``(n_steps, n_edges)``
        Pre-computed occupancy ratios in [0, 1] (e.g. from
        :func:`compute_occupancy`).
    edges_df : DataFrame
        Edges table.
    edge_indices : list of int or None
        Subset of edge indices to plot.  If *None*, all edges are shown.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # --- time range slicing ---
    _t0 = t_start if t_start is not None else 0
    _t1 = t_end if t_end is not None else occupancy.shape[0]
    t_offset = _t0
    occupancy = occupancy[_t0:_t1]

    n_steps = occupancy.shape[0]
    n_edges = len(edges_df)
    if edge_indices is None:
        edge_indices = list(range(n_edges))

    y_values = occupancy * 100.0

    fig = go.Figure()
    from_col = edges_df["from"].values.astype(int)
    to_col = edges_df["to"].values.astype(int)
    for ei in edge_indices:
        u, v = int(from_col[ei]), int(to_col[ei])
        fig.add_trace(go.Scatter(
            x=list(range(t_offset, t_offset + n_steps)), y=y_values[:, ei],
            mode="lines", name=f"edge {ei} ({u}\u2192{v})",
        ))
    fig.update_layout(
        xaxis_title="Time step", yaxis_title="Occupancy (%)",
        template="simple_white",
    )
    return fig


__all__ = [
    "plot_network",
    "animate",
    "animate_mpl",
    "animate_map",  # deprecated
    "animate_occupancy",
    "animate_occupancy_map",  # deprecated
    "animate_occupancy_mpl",
    "compute_occupancy",
    "compute_edge_capacity",
    "from_osmnx",
    "plot_edge_occupancy",
    "OSMnxResult",
]
