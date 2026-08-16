_base_ = './baseline.py'
OptimizationParams = dict(
    residual_densify = True,
    residual_densify_topk = 0.01,
    residual_densify_max_points = 3000,
)
