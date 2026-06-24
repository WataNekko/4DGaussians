#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
import copy
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
from scene.cameras import Camera


def generate_custom_path(scene, num_frames=300):
    train_cams = scene.getTrainCameras()

    # 1. Select Keyframes
    # Let's automatically pick 5 evenly spaced cameras to act as our path anchors.
    # Alternatively, you can manually select specific camera indices you like.
    num_keyframes = 5
    step = len(train_cams) // num_keyframes
    print("step", step)
    keyframe_cams = [
        train_cams[min(i * step, len(train_cams) - 1)] for i in range(num_keyframes)
    ]

    # We map these keyframes to specific "times" along our path (0.0 to 1.0)
    key_times = np.linspace(0.0, 1.0, len(keyframe_cams))

    Rs = []
    Ts = []
    Centers = []
    is_dict = isinstance(keyframe_cams[0], dict)

    # 2. Extract Data from all Keyframes
    for cam in keyframe_cams:
        R = cam["R"] if is_dict else cam.R
        T = cam["T"] if is_dict else cam.T

        Rs.append(R)
        Ts.append(T)
        # Calculate World Center: C = -R^T * T
        Centers.append(-np.dot(R.T, T))

    # 3. Set up multi-point Interpolators
    # Slerp natively accepts an array of times and a sequence of Rotations
    slerp_operator = Slerp(key_times, R_scipy.from_matrix(Rs))

    # CubicSpline creates a smooth curved 3D path through our World Centers
    spline_operator = CubicSpline(key_times, Centers, bc_type="natural")

    custom_cameras = []
    cam_template = keyframe_cams[0]  # Use the first camera as our blueprint

    for i in range(num_frames):
        progress = i / float(num_frames - 1)

        # Interpolate Rotation and Position for the current frame
        r_interp = slerp_operator(progress).as_matrix()
        pos_interp = spline_operator(progress)

        # Convert interpolated World Center back to Translation vector: T = -R * C
        t_interp = -np.dot(r_interp, pos_interp)

        if is_dict:
            # Dictionaries don't have pre-baked tensors, so deepcopy works here
            new_cam = copy.deepcopy(cam_template)
            new_cam["R"] = r_interp
            new_cam["T"] = t_interp
            if "timestamp" in new_cam:
                new_cam["timestamp"] = progress
            if "time" in new_cam:
                new_cam["time"] = progress
            custom_cameras.append(new_cam)

        else:
            # For actual Camera objects, we MUST instantiate a new one
            # so the W2C and Projection tensors are recalculated.
            current_time = progress

            # 1. Initialize with ONLY standard 3DGS arguments
            new_cam = Camera(
                colmap_id=cam_template.colmap_id,
                R=r_interp,
                T=t_interp,
                FoVx=cam_template.FoVx,
                FoVy=cam_template.FoVy,
                image=cam_template.original_image,
                gt_alpha_mask=None,
                image_name=f"custom_frame_{i}",
                uid=i,
                data_device=cam_template.data_device,
            )

            # 2. Dynamically attach the 4D time variables AFTER initialization
            new_cam.timestamp = current_time
            new_cam.time = current_time
            new_cam.fid = current_time

            custom_cameras.append(new_cam)

    return custom_cameras

    return custom_cameras


def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(
                image, os.path.join(path, "{0:05d}".format(count) + ".png")
            )
            return count, True
        except:
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)


to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def render_set(
    model_path, name, iteration, views, gaussians, pipeline, background, cam_type
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print("point nums:", gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:
            time1 = time()

        rendering = render(view, gaussians, pipeline, background, cam_type=cam_type)[
            "render"
        ]
        render_images.append(to8b(rendering).transpose(1, 2, 0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt = view["image"].cuda()
            gt_list.append(gt)

    time2 = time()
    print("FPS:", (len(views) - 1) / (time2 - time1))

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    imageio.mimwrite(
        os.path.join(model_path, name, "ours_{}".format(iteration), "video_rgb.mp4"),
        render_images,
        fps=30,
    )


def render_sets(
    dataset: ModelParams,
    hyperparam,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    skip_video: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type = scene.dataset_type
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
            )
        if not skip_video:
            # --- ADD YOUR CUSTOM RENDER BLOCK HERE ---
            print("Generating custom Slerp trajectory...")
            custom_path = generate_custom_path(scene, num_frames=100)

            # We pass "custom_video" as the name so it creates a separate folder
            render_set(
                dataset.model_path,
                "custom_video",
                scene.loaded_iter,
                custom_path,
                gaussians,
                pipeline,
                background,
                cam_type,
            )
            # render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
    )
