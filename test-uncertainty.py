import numpy as np
import pandas as pd
from fts import Simulator

def run():
    edges_file = 'data/demo_graph_adj.csv'
    vehicles_file = 'data/demo_demand.csv'
    edges = pd.read_csv(edges_file, names=['id', 'from', 'to', 'length', 'speed', 'lanes'],
                        skiprows=1, index_col='id')
    vehicles = pd.read_csv(vehicles_file, names=['origin', 'destination', 'start'],
                           skiprows=1)
    vehicles['start'] = vehicles['start'].astype(int)
    vehicles.index.name = 'id'
    simulator = Simulator(vehicles=vehicles, edges=edges)
    step = 0
    logs = []

    log_step = np.vstack([simulator._vehicles.edge, simulator._vehicles.edge_distance])
    log_step[1, simulator._vehicles.edge == -1] = 0
    log_step = log_step.swapaxes(0, 1)
    logs.append(log_step)

    while True:
        simulator.step(False)
        step += 1
        nr_waiting = (simulator._vehicles.status == Simulator.VehicleStatus.WAITING.value).sum()
        nr_at_node = (simulator._vehicles.status == Simulator.VehicleStatus.AT_NODE.value).sum()
        nr_in_edge = (simulator._vehicles.status == Simulator.VehicleStatus.IN_EDGE.value).sum()
        nr_arrived = (simulator._vehicles.status == Simulator.VehicleStatus.ARRIVED.value).sum()

        log_step = np.vstack([simulator._vehicles.edge, simulator._vehicles.edge_distance])
        log_step[1, simulator._vehicles.edge == -1] = 0
        log_step = log_step.swapaxes(0, 1)
        logs.append(log_step)

        if nr_waiting + nr_at_node + nr_in_edge == 0:
            break

    logs = np.array(logs)

    return logs

def vehicle_history(logs, v):
    print(f"History of vehicle {v}")
    edge = -1
    for t in range(logs.shape[0]):
        old_edge = edge
        edge = int(logs[t,v,0])
        dist = logs[t,v,1]
        if edge != old_edge:
            if edge == -1:
                print(f"- Time {t}: vehicle {v} arrives")
            else:
                print(f"- Time {t}: vehicle {v} enters edge {edge}")
        if edge != -1:
            print(f"- Time {t}: vehicle {v} is at distance {dist} in edge {edge}")

def edge_history(logs, edge):
    print(f"History of edge {edge}")
    vehicles = set()
    for t in range(logs.shape[0]):
        old_vehicles = vehicles
        vehicles = set()
        for v in range(logs.shape[1]):
            if logs[t,v,0] == edge:
                vehicles.add(v)
                if v not in old_vehicles:
                    print(f"- Time {t}: vehicle {v} enters edge {edge}")
                dist = logs[t,v,1]
                print(f"- Time {t}: vehicle {v} is at distance {dist} in edge {edge}")
        for v in old_vehicles - vehicles:
            print(f"- Time {t}: vehicle {v} leaves edge {edge}")

if __name__ == "__main__":
    logs = run()
    vehicle_history(logs, 3)
    edge_history(logs, 7)
