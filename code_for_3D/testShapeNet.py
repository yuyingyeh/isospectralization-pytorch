"""Reconstruct a shape from a portion of eigenvalue sequence."""
from shape_library import load_mesh, prepare_mesh, save_ply
from spectrum_alignment import OptimizationParams, calc_evals, run_optimization

params = OptimizationParams()
params.checkpoint_steps = 100
params.eval_steps = 100
params.min_eval_loss = 0.0001
params.evals = [20]
params.steps = 3000

#shapeName = 'ShapeNet1_bottle'
#shapeName = 'ShapeNet2_bowl'
#shapeName = 'ShapeNet3_sp_bottle'
#shapeName = 'ShapeNet4_sp_airplane'
#shapeName = 'ShapeNet5_sp_bus'
#shapeName = 'ShapeNet6_sp_can'
#shapeName = 'ShapeNet7_sp_can'
#shapeName = 'ShapeNet8_bowl_bowl'
#shapeName = 'ShapeNet9_jar_jar'
#shapeName = 'ShapeNet10_tub_tub'
shapeName = 'ShapeNet11_bottle_bottle'

VERT, TRIV = load_mesh("data/%s/modelS/" % shapeName)
mesh = prepare_mesh(VERT, TRIV, "float32")

VERT_t, TRIV_t = load_mesh("data/%s/modelT/" % shapeName)
evals_t = calc_evals(VERT_t, TRIV_t)

out_path = "results/%s_out_large_vol_coef" % shapeName

run_optimization(
    mesh=mesh,
    target_evals=evals_t,
    out_path=out_path,
    params=params,
)

# Save the target embedding
save_ply(VERT_t, TRIV_t, "%s/ply/target.ply" % out_path)
