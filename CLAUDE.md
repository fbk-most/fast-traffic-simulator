# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fast Traffic Simulator (FTS) is a NumPy/SciPy-based microscopic traffic simulator for directed road networks. Vehicles travel shortest paths (Dijkstra) through a directed graph, following Newell's car-following model. Developed at Fondazione Bruno Kessler (@fbk-most).

**Status:** Early alpha — APIs may change without notice.

## Development Setup

```bash
uv sync                  # install dependencies
uv sync --group dev      # include dev tools (mypy, stubs)
```

Build system: Hatchling. Python >=3.12 required.

## Type Checking

```bash
uv run mypy fts/
```

## Running Examples

```bash
uv run python examples/demo.py
uv run python examples/visualize.py
```

## Architecture

The package has two modules under `fts/`:

- **`simulator.py`** — Core `Simulator` class. All state is stored in flat NumPy arrays inside two dataclasses: `VehiclesRecord` (per-vehicle state) and `EdgesRecord` (per-edge state). The simulation loop is a single `step()` method that processes phases in order: starting → out-of-edge → arrived → entering-edge → progress. Routing uses `scipy.sparse.csgraph.dijkstra` with inverted from/to to get successor nodes. Entry point is `Simulator.build()` which returns `(simulator, fixed_vehicles)`.

- **`visualization.py`** — Plotly and Matplotlib animation/plotting utilities. Two rendering backends: Plotly (`animate`, `animate_occupancy`) for interactive HTML, and Matplotlib (`animate_mpl`, `animate_occupancy_mpl`) for mp4/gif export. `from_osmnx()` imports real road networks from OpenStreetMap. All functions take the edges DataFrame and a `pos` dict mapping node IDs to (x,y) coordinates.

## Key Design Decisions

- All vehicle/edge state uses flat NumPy arrays (no per-vehicle objects) for vectorized performance
- Lane assignment: vehicle enters the lane with the most available space; no lane changes after entry
- Nodes are dimensionless — travel time accrues only on edges
- `last_vehicle` array in EdgesRecord uses sentinel values: `-1` = empty lane, `-2` = lane doesn't exist
- `step(update_next_leg=True)` recomputes routes using instantaneous speeds (expensive, use sparingly)

## Dependencies

Core: numpy, scipy, pandas, pyarrow, plotly, networkx, salib, dash, osmnx
Optional (`map` extra): osmnx, matplotlib, contextily (for map tile backgrounds)