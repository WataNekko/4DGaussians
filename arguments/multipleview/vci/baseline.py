ModelParams = dict(
    held_out_cams = "3",
    test_iterations=[3000, 5000, 7000, 10000, 14000, 16000, 20000],
    save_iterations=[7000, 10000, 14000, 16000, 20000],
)
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4,8],
    defor_depth = 1,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False,
    white_background=False
)
OptimizationParams = dict(
    dataloader=True,
    iterations = 20000,
    batch_size=4,
    coarse_iterations = 5000,
    densify_until_iter=15000,
)
