
from sensitivity_analysis_framework import run_sensitivity_analysis, build_dash_app

import numpy as np

def simulator_fn(
    params: dict,
    series_axis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    a = params["a"]
    b = params["b"]
    c = params["c"]
    Y = a + series_axis*series_axis*b*c
    return Y

SA_PROBLEM = {
    "num_vars": 3,
    "names": ["a", "b", "c"],
    "bounds": [[0, 1], [0, 1], [0, 1]],
}

N_SAMPLES = 512
N_REPLICAS = 1
SEED = 42

SERIES_AXIS = np.linspace(-2,2,200)

results = run_sensitivity_analysis(
    simulator_fn=simulator_fn,
    problem=SA_PROBLEM,
    series_axis=SERIES_AXIS,
    n_samples=N_SAMPLES,
    n_replicas=N_REPLICAS,
    seed=SEED,
    analyze_variance=False,
)

app = build_dash_app(results)
app.run(debug=False, port=6969)