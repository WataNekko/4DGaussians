ModelParams = dict(
    # held_out_cams = "3,11,22",
)

ModelHiddenParams = dict(
    # ============ Hexplane (K-Planes) spatio-temporal feature grid ============
    # This is the underlying representation the deformation MLP reads from.
    kplanes_config = {
        'grid_dimensions': 2,       # pairwise planes (xy, xz, yz, xt, yt, zt) -- standard, don't change
        'input_coordinate_dim': 4,  # x, y, z, t -- don't change
        'output_coordinate_dim': 32,# feature channels per queried point. Higher = more capacity to
                                     # represent complex motion/appearance, but more memory + slower.
                                     # 16 (repo default) -> 32 is a reasonable bump for a real capture
                                     # with more complex motion than the synthetic benchmarks.
        'resolution': [64, 64, 64, 25]
        # first 3 numbers: spatial grid resolution (same in x/y/z here).
        # last number: temporal grid resolution -- repo's own guidance is "about half the number
        # of dynamic frames you have". If your capture is e.g. 150 frames, 75 is more apt than 25;
        # if it's ~50 frames, 25 is fine. Too low = can't represent fast local motion; too high =
        # slower and more prone to overfitting/noise.
    },
    multires = [1, 2],
    # Multi-resolution voxel grid used by the coarse-stage (static) branch. Each entry is a
    # downsampling factor; more entries = more coarse-to-fine capacity for the initial static
    # reconstruction, at the cost of memory. [1,2] is the repo's lighter default; [1,2,4,8] is
    # the "full" version used in some example configs -- try [1,2,4,8] if the coarse stage
    # (the initial rough static reconstruction, before motion is learned) looks under-detailed.

    defor_depth = 0,   # depth of the deformation MLP (0 = a single layer). Increase (try 1-2) if
                       # the fine stage's motion looks under-fit (e.g. rigid parts not moving
                       # enough); each +1 meaningfully slows training and rendering.
    net_width = 128,   # width of the deformation MLP. Wider = more capacity, slower, more prone
                       # to overfitting with a modest number of training views.

    # ============ Regularization of the Hexplane feature grid itself ============
    # (NOT per-Gaussian trajectory smoothness -- that's the new traj_smooth_* below.)
    plane_tv_weight = 0.0002,        # total-variation smoothness on the 3 *spatial* planes.
    time_smoothness_weight = 0.001,  # total-variation smoothness on the 3 *spatio-temporal* planes.
    l1_time_planes = 0.0001,         # L1 sparsity penalty on the spatio-temporal planes.
    # All three regularize the underlying feature grid so it doesn't overfit noise. Too low and
    # you may see grid-induced jitter; too high and fine motion gets smoothed away. These are
    # already small (repo default) -- don't crank these up as your "temporal smoothness fix";
    # use traj_smooth_weight below instead, which is a different, more targeted mechanism.

    no_do = False,     # False = ALLOW opacity to be deformed over time (repo default for this
                       # is actually True/off -- I'm turning it ON here since your captures likely
                       # have real appearance/visibility changes over time, e.g. objects entering/
                       # leaving occlusion, which a static per-point opacity can't represent).
    no_dshs = False,   # same idea but for the SH color coefficients (allow color/shading to change
                       # over time, e.g. specular highlights moving) -- also repo-default True/off.
    no_ds = False,     # allow scaling to deform (usually left False/allowed).
    no_dx = False,     # allow position to deform -- essential for a dynamic scene, always False.
    no_dr = False,     # allow rotation to deform -- usually left False/allowed.
    no_grid = False,   # False = use the Hexplane at all. Never set True unless debugging.
    empty_voxel = False,   # author-flagged unused, leave alone.
    grid_pe = 0,           # author-flagged unused, leave alone.
    static_mlp = False,    # author-flagged unused, leave alone.
    apply_rotation = False,# author-flagged unused, leave alone.

    # ============ NEW: trajectory smoothness (acceleration penalty) ============
    traj_smooth_weight = 0.01,
    # Weight on the new compute_trajectory_smoothness_loss. 0 = off. This penalizes jittery/
    # inconsistent per-Gaussian motion (2nd derivative of position) without penalizing genuine
    # smooth motion -- unlike a naive render(t) vs render(t+dt) loss. Start small (0.001-0.01)
    # and sweep upward; too high can still over-rigidify motion (fast-moving parts start to lag).
    traj_smooth_dt = 0.02,
    # Time offset used for the 3-point finite difference (t-dt, t, t+dt). Should be small relative
    # to your frame spacing (frame spacing in normalized time is ~1/num_frames) -- 0.02 is
    # reasonable for ~50-150 frame captures; shrink it for very long sequences.
    traj_smooth_sample_size = 20000,
    # Subsamples this many "dynamic" Gaussians per call for speed (querying the deformation MLP
    # 3x for every single dynamic point every iteration would be slow for large point clouds).
    # Set to None to use all dynamic points if you have GPU headroom and want the exact loss.

    random_background = False,
    white_background = False,   # keep this matching how you preprocessed your masks
    eval = True,                # holds out a test set -- combine with held_out_cams below
)

OptimizationParams = dict(
    dataloader = True,   # use torch DataLoader-based batching -- recommended for 35 cameras
                          # worth of data (repo default is actually False; True helps I/O
                          # throughput on larger datasets like yours).
    iterations = 30_000,
    coarse_iterations = 3000,   # iterations spent on the static/coarse warm-up before the fine
                                # (motion-aware) stage begins. Increase if the coarse-stage point
                                # cloud still looks noisy/incomplete when fine stage starts.
    batch_size = 1,             # cameras rendered per training step. Bumping this smooths gradient
                                # estimates (uses more views per step) but increases memory/step time.

    # ============ Densification / pruning schedule (existing, unchanged mechanics) ============
    densify_from_iter = 500,          # densification starts after this many iterations
    densify_until_iter = 15_000,      # and stops after this many (fine stage keeps training after,
                                       # but point count is frozen from here on)
    densification_interval = 100,     # how often (in iterations) a densify() pass runs
    densify_grad_threshold_coarse = 0.0002,     # gradient magnitude above which a point is a
    densify_grad_threshold_fine_init = 0.0002,  # clone/split candidate. Lower = more aggressive
    densify_grad_threshold_after = 0.0002,      # densification (more points, more floater risk).
                                                 # fine_init->after interpolates over the fine stage.
    pruning_from_iter = 500,
    pruning_interval = 100,
    opacity_threshold_coarse = 0.005,       # points with opacity below this get pruned. Lower =
    opacity_threshold_fine_init = 0.005,    # keeps weaker points around longer (risk: floaters
    opacity_threshold_fine_after = 0.005,   # linger); higher = prunes more aggressively (risk:
                                             # losing thin/faint real structure too early).
    opacity_reset_interval = 3000,   # every N iterations, ALL opacities get reset to a small value
                                      # (classic 3DGS anti-floater trick -- forces every point to
                                      # re-earn its opacity through the photometric loss). Also
                                      # gates when screen-size-based pruning kicks in.
    percent_dense = 0.01,            # a Gaussian whose scale exceeds this fraction of the scene
                                      # extent is treated as "too big" -> split candidate instead
                                      # of clone candidate.

    # ============ Loss weights ============
    lambda_dssim = 0.2,   # repo default is 0 (pure L1). Since you're already reporting SSIM/LPIPS,
                           # it's reasonable to optimize partly for structural similarity too --
                           # 0.2 is a common starting point (used in the original 3DGS paper).
                           # This is a baseline hyperparameter choice, not one of your 3 techniques
                           # -- tune/ablate it separately from them.
    lambda_lpips = 0,      # off by default; LPIPS-in-the-loop is expensive (needs a VGG/AlexNet
                           # forward+backward every step) -- leave off unless you have time budget.

    # ============ NEW: camera-holdout split ============
    # (this field actually lives on ModelParams, not here -- pass it as a CLI flag instead:
    #  --held_out_cams 3,11,22   -- can't be set from this dict since ModelParams isn't merged
    #  the same way; see note below.)

    # ============ NEW: confidence-aware pruning ============
    min_visibility_count = 2,
    # A point seen by fewer than this many distinct training views during the last
    # pruning_interval window gets pruned, in addition to the existing opacity/screen-size
    # checks. 0 disables it (repo behavior). Start at 2; raise to 3-4 if floaters persist,
    # but watch for real-but-rarely-seen thin structures getting pruned too eagerly if you
    # go much higher.

    # ============ NEW: residual-guided densification ============
    residual_densify = True,
    residual_densify_topk = 0.005,
    # Fraction of pixels (by highest photometric error) to spawn new Gaussians from, per event.
    # 0.005 = top 0.5% of pixels in the reference view. Raise if blur persists in large regions;
    # lower if point count grows too fast / runs out of memory.
    residual_densify_max_points = 2000,
    # Hard cap on new points injected per event, regardless of topk fraction -- a safety valve
    # so a single bad frame can't blow up your point budget.
)
