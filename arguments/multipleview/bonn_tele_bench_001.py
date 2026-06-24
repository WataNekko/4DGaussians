ModelHiddenParams = dict(
    # Point this to the root of your newly structured dataset
    source_path = "data/multipleview/bonn_tele_bench_001",
    
    # Where the trained splats and renders will be saved
    model_path = "output/bonn_tele_bench_001",
    
    # Set to True ONLY if you pre-multiplied your masks with a white background. 
    # If you multiplied with black (or left the background as-is), keep this False.
    white_background = False,
    
    # Assuming your symlinked frames are .jpg
    extension = ".jpg",
    
    # Random background color if white_background is False
    random_background = False,
    
    # Evaluates the model by holding out a default test set of cameras/frames
    eval = True 
)

OptimizationParams = dict(
    # 4DGS defaults to 40,000, but 30,000 is often enough for a multi-view baseline
    iterations = 30_000,
    
    # When to start optimizing the deformation network (temporal aspect)
    coarse_iterations = 3000, 
    
    # When to stop densifying the point cloud
    densify_until_iter = 15_000, 
    
    # Use voxel deformation for the dynamic aspects
    voxel_deform = True
)

PipelineParams = dict(
    convert_SHs_python = False,
    compute_cov3D_python = False
)
