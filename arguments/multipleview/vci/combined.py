_base_ = './baseline.py'
ModelHiddenParams = dict(
    traj_smooth_weight = 0.01,
    traj_smooth_dt = 0.02,
    traj_smooth_sample_size = 20000,
)
OptimizationParams = dict(
    min_visibility_count = 5,
    residual_densify = True,
    residual_densify_topk = 0.005,
    residual_densify_max_points = 2000,
)
