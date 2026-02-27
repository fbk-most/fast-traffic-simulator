"""Minimal quickstart example: simulate one vehicle on a synthetic two-edge graph.

No input files are required — the network and demand are built in memory.

Usage::

    uv run python examples/quickstart.py
"""

import pandas as pd
from fts import Simulator

# Build a minimal two-edge graph: 0 → 1 → 2
edges = pd.DataFrame({
    'from':   [0, 1],
    'to':     [1, 2],
    'length': [100.0, 150.0],
    'speed':  [10.0, 15.0],
    'lanes':  [1, 1],
})

# One vehicle travelling from node 0 to node 2, departing at step 0
vehicles = pd.DataFrame({
    'origin':      [0],
    'destination': [2],
    'start':       [0],
})

simulator, _ = Simulator.build(edges=edges, vehicles=vehicles)

while True:
    simulator.step()
    arrived = (simulator.vehicles.status == Simulator.VehicleStatus.ARRIVED.value).sum()
    if arrived == len(vehicles):
        break

print(f"Vehicle arrived at step {simulator.vehicles.arrival_time[0]}")
