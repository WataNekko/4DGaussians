_base_ = './baseline.py'
ModelHiddenParams = dict(
    traj_smooth_weight = 0.002,
    traj_smooth_dt = 0.02,
    traj_smooth_sample_size = 20000,
)
