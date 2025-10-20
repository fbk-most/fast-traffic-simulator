import time
from legacy import *
from fts import Simulator

if __name__ == '__main__':
    # Emit verbose diagnostics in building phase
    VERBOSE = False
    # Switch between randomized and deterministic behavior of the simulator
    RANDOM = False
    # Shortest path recomputation interval
    SP = 300
    # Simulation scenario
    # SCENARIO = "Platoons/Highways"
    SCENARIO = "Vehicles/Highways"
    # SCENARIO = "Platoons/AllRoads"
    # SCENARIO = "Vehicles/AllRoads"

    print(f"Scenario: {SCENARIO} - Random: {RANDOM} - SP recomputation: {SP} seconds")

    # Load data
    print("Loading data...")
    start_load = time.time()

    files = {
        "Platoons/Highways": ['nodes__v3.csv', 'edges__v3.csv', 'platoons_size=20_reduction=0.0__v3_seed_0.parquet'],
        "Vehicles/Highways": ['nodes__v3.csv', 'edges__v3.csv', 'platoons_size=1_reduction=0.0__v3_seed_0.parquet'],
        "Platoons/AllRoads": ['nodes__allRoads_v2.csv', 'edges__allRoads_v2.csv',
                              'platoons_size=20_reduction=0.0__allRoads_v2_seed_0.parquet'],
        "Vehicles/AllRoads": ['nodes__allRoads_v2.csv', 'edges__allRoads_v2.csv',
                              'platoons_size=1_reduction=0.0__allRoads_v2_seed_0.parquet']}
    directory = "data"

    (nodes, edges, vehicles) = read_legacy(*(directory + "/" + f for f in files[SCENARIO]))

    # TODO: Manage multi-edges
    n_duplicated = edges.duplicated(subset=['from', 'to']).sum()
    if n_duplicated > 0:
        print(f"*** Removing {n_duplicated} multiple edges ***")
        if VERBOSE:
            duplicated = edges[edges.duplicated(subset=['from', 'to'], keep=False)].groupby(['from', 'to'])
            for d,_ in duplicated:
                print("  * Edge cluster: [",
                      ", ".join([str(e) for e in d]),
                      "]")
        edges.drop_duplicates(subset=['from', 'to'], inplace=True)
    load_time = time.time() - start_load
    print(f"Data loaded in {load_time:.2f} seconds")

    # For quick tests, the number of vehicles can be reduced and the start time can be lowered
    # vehicles = vehicles.head(100000)
    ### vehicles['start'] = 0

    # Convert data
    print("Converting data...")
    start_convert = time.time()
    nodes_map = map_nodes(nodes)
    converted_edges = convert_edges(edges, nodes_map)
    converted_vehicles = convert_vehicles(vehicles, nodes_map)
    convert_time = time.time() - start_convert
    print(f"Data converted in {convert_time:.2f} seconds")

    # Initialize the simulator
    print("Initializing simulator...")
    start_init = time.time()
    simulator = Simulator(vehicles=converted_vehicles, edges=converted_edges, random=RANDOM)
    # TODO: Manage unreachable destinations!
    destination_eq_origin = (simulator._vehicles.destination == simulator._vehicles.origin)
    if destination_eq_origin.any():
        print(f"*** Detecting {destination_eq_origin.sum()} vehicles with destination equal to origin ***")
        if VERBOSE:
            for v in destination_eq_origin.nonzero()[0]:
                print("  * Vehicle:",
                      f"{vehicles.index[v]}: {vehicles.iloc[v]['origin']} -> {vehicles.iloc[v]['destination']}")
    unreachable = ((simulator._vehicles.destination != simulator._vehicles.origin) &
                   (simulator._next_leg[simulator._vehicles.destination, simulator._vehicles.origin] == -9999))
    if unreachable.any():
        print(f"*** Rerouting {unreachable.sum()} vehicles with unreachable destinations ***")
        if VERBOSE:
            for v in unreachable.nonzero()[0]:
                print("  * Vehicle:",
                      f"{vehicles.index[v]}: {vehicles.iloc[v]['origin']} -> {vehicles.iloc[v]['destination']}")
        simulator._vehicles.destination[unreachable] = simulator._vehicles.origin[unreachable]

    init_time = time.time() - start_init
    print(f"Simulator initialized in {init_time:.2f} seconds")

    # Run simulation
    print("Starting simulation...")
    # Main timing
    start_sim = time.time()
    h = 0
    while True:
        for s in range(60*60):
            simulator.step(s % SP == SP-1)
        h += 1
        nr_waiting = (simulator._vehicles.status == Simulator.VehicleStatus.WAITING.value).sum()
        nr_at_node = (simulator._vehicles.status == Simulator.VehicleStatus.AT_NODE.value).sum()
        nr_in_edge = (simulator._vehicles.status == Simulator.VehicleStatus.IN_EDGE.value).sum()
        nr_arrived = (simulator._vehicles.status == Simulator.VehicleStatus.ARRIVED.value).sum()
        print(f"... simulation time after {h} hours: {time.time()-start_sim:.2f} seconds")
        print(f"...... waiting: {nr_waiting} vehicles")
        print(f"...... at node: {nr_at_node} vehicles")
        print(f"...... in edge: {nr_in_edge} vehicles")
        print(f"...... arrived: {nr_arrived} vehicles")

        if nr_waiting + nr_at_node + nr_in_edge == 0:
            break

    sim_time = time.time() - start_sim
    total_steps = h*60*60
    steps_per_second = total_steps / sim_time

    print("\n--- Performance Summary ---")
    print(f"Total simulation time: {sim_time:.2f} seconds")
    print(f"Steps per second: {steps_per_second:.2f}")
    print(f"Time per step: {(sim_time/total_steps)*1000:.4f} ms")
    print(f"Total runtime: {load_time + convert_time + init_time + sim_time:.2f} seconds")

    travel_times = simulator._vehicles.arrival_time - converted_vehicles['start'].values
    print("\n--- Travel Statistics ---")
    print(f"Average travel time: {travel_times.mean():.2f} seconds")
    print(f"Minimum travel time: {travel_times.min():.2f} seconds")
    print(f"Maximum travel time: {travel_times.max():.2f} seconds")
