"""Reconstruct a shape from a portion of eigenvalue sequence."""
from shape_library import load_mesh, prepare_mesh, resample
from spectrum_alignment import OptimizationParams, calc_evals, run_optimization

params = OptimizationParams()
params.evals = [30]
params.min_eval_loss = 0.05
params.steps = 5000
params.plot = False

[VERT, TRIV] = load_mesh("data/oval/")
[VERT, TRIV] = resample(VERT, TRIV, 300)

[VERT_t, TRIV_t] = load_mesh("data/bell/")
evals_t = calc_evals(VERT_t, TRIV_t)
mesh = prepare_mesh(VERT, TRIV, "float32")
run_optimization(
    mesh=mesh, target_evals=evals_t, out_path="results/bell", params=params
)
