"""Demo script: run the simulator on a small graph and trace vehicle/edge histories.

Usage::

    uv run python examples/demo.py

Input files are located in ``examples/demo_data/``.
"""

import numpy as np
import pandas as pd
from fts import Simulator


def run() -> np.ndarray:
    """Run the demo simulation and return per-step position logs.

    Returns:
        A float array of shape ``(steps, nr_vehicles, 2)`` where the last axis
        holds ``[edge_index, edge_distance]`` for each vehicle at each step.
        Edge index is ``-1`` (distance set to 0) when the vehicle is not in an
        edge.
    """
    edges_file = 'examples/demo_data/demo_graph.csv'
    vehicles_file = 'examples/demo_data/demo_demand.csv'
    edges = pd.read_csv(edges_file, names=['id', 'from', 'to', 'length', 'speed', 'lanes'],
                        skiprows=1, index_col='id')
    vehicles = pd.read_csv(vehicles_file, names=['origin', 'destination', 'start'],
                           skiprows=1)
    vehicles['start'] = vehicles['start'].astype(int)
    vehicles.index.name = 'id'

    simulator, _ = Simulator.build(edges=edges, vehicles=vehicles)
    logs = []

    def snapshot() -> np.ndarray:
        log = np.vstack([simulator.vehicles.edge, simulator.vehicles.edge_distance])
        log[1, simulator.vehicles.edge == -1] = 0
        return log.swapaxes(0, 1)

    logs.append(snapshot())

    while True:
        simulator.step(False)
        nr_waiting = (simulator.vehicles.status == Simulator.VehicleStatus.WAITING.value).sum()
        nr_at_node = (simulator.vehicles.status == Simulator.VehicleStatus.AT_NODE.value).sum()
        nr_in_edge = (simulator.vehicles.status == Simulator.VehicleStatus.IN_EDGE.value).sum()
        logs.append(snapshot())

        if nr_waiting + nr_at_node + nr_in_edge == 0:
            break

    return np.array(logs)


def vehicle_history(logs: np.ndarray, v: int) -> None:
    """Print a human-readable movement history for a single vehicle.

    Args:
        logs: Position log array as returned by :func:`run`.
        v: Vehicle index.
    """
    print(f"History of vehicle {v}")
    edge = -1
    for t in range(logs.shape[0]):
        old_edge = edge
        edge = int(logs[t, v, 0])
        dist = logs[t, v, 1]
        if edge != old_edge:
            if edge == -1:
                print(f"- Time {t}: vehicle {v} arrives")
            else:
                print(f"- Time {t}: vehicle {v} enters edge {edge}")
        if edge != -1:
            print(f"- Time {t}: vehicle {v} is at distance {dist} in edge {edge}")


def edge_history(logs: np.ndarray, edge: int) -> None:
    """Print a human-readable occupancy history for a single edge.

    Args:
        logs: Position log array as returned by :func:`run`.
        edge: Edge index.
    """
    print(f"History of edge {edge}")
    vehicles: set[int] = set()
    for t in range(logs.shape[0]):
        old_vehicles = vehicles
        vehicles = {v for v in range(logs.shape[1]) if logs[t, v, 0] == edge}
        for v in vehicles:
            dist = logs[t, v, 1]
            if v not in old_vehicles:
                print(f"- Time {t}: vehicle {v} enters edge {edge}")
            print(f"- Time {t}: vehicle {v} is at distance {dist} in edge {edge}")
        for v in old_vehicles - vehicles:
            print(f"- Time {t}: vehicle {v} leaves edge {edge}")


if __name__ == "__main__":
    logs = run()
    vehicle_history(logs, 3)
    edge_history(logs, 7)
