# Fast Traffic Simulator

A NumPy/SciPy-based microscopic traffic simulator for directed road networks.

We develop this package at [@fbk-most](https://github.com/fbk-most), a research unit
at [Fondazione Bruno Kessler](https://www.fbk.eu/en/).

The simulator models individual vehicles travelling from an origin node to a
destination node across a directed graph of road edges.  Shortest-path routing
is computed via Dijkstra's algorithm and can be refreshed periodically to
reflect current congestion.  Vehicle behaviour on links follows Newell's
simplified car-following model: each vehicle moves as fast as possible while
respecting the road speed limit and keeping a safe distance from the vehicle
ahead.  This reproduces queue formation and congestion without explicit capacity
constraints.  Nodes are treated as dimensionless transfer points — travel time
accrues only on links — and each edge supports multiple independent lanes.

*Note: this package is currently in an early development stage. APIs may change
without notice between releases.*

## Installation

```bash
pip install fast-traffic-simulator
```

Or, for development (requires [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/fbk-most/fast-traffic-simulator
cd fast-traffic-simulator
uv sync
```

## Quick start

```python
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
```

## Main contributors

[Marco Pistore](https://it.linkedin.com/in/marco-pistore): design and implementation.

## Acknowledgements

This software has been developed in the scope of the Bologna Digital Twin project, 
and partially supported by the following projects:

- NRR ICSC National Research Centre for High Performance Computing, Big Data and Quantum Computing (CN00000013), under
  the NRRP MUR program funded by the NextGenerationEU.
- European Structural and Investment Funds, as part of the National Program for Metropolitan Cities and Medium-Sized
  Cities South 2021-2027, Priority 1 Digital Agenda and Urban Innovation,Action 1.1.2.1 Metropolitan Digital Agenda,
  Project BO1.1.2.1.a "DIGITAL TWIN: GOVERNANCE AND ENHANCEMENT OF DATA ASSETS"

## License

Copyright 2025-2026 Fondazione Bruno Kessler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

```
SPDX-License-Identifier: Apache-2.0
```
