import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split,
        held_out_cams=None
    ):
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height=cam_intrinsics[1].height
        width=cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.held_out_cams = held_out_cams or []
        self.image_paths, self.image_poses, self.image_times= self.load_images_path(cam_folder, cam_extrinsics,cam_intrinsics,split)
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)


    def load_images_path(self, cam_folder, cam_extrinsics,cam_intrinsics,split):
        images = os.listdir(os.path.join(cam_folder,"cam01"))
        ext = images[0].split(".")[1]
        image_length = len(images)
        image_paths=[]
        image_poses=[]
        image_times=[]
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            number = os.path.basename(extr.name)[5:-4]
            images_folder=os.path.join(cam_folder,"cam"+number.zfill(2))
            is_held_out = int(number) in self.held_out_cams

            if self.held_out_cams:
                # Camera-holdout mode: a held-out camera contributes ALL its frames to
                # "test" (so you get a full video for temporal/video metrics) and NONE
                # to "train". A training camera contributes ALL its frames to "train"
                # and NONE to "test" -- so train/test are mutually exclusive by camera,
                # no leakage between the splits.
                if split == "test" and not is_held_out:
                    continue
                if split == "train" and is_held_out:
                    continue
                image_range = range(image_length)
            else:
                # Legacy behaviour (kept for backward compatibility with other
                # multipleview-format datasets that don't use camera holdout):
                # every camera contributes to both splits, test uses 3 sparse frames.
                # NOTE: this means train and test overlap in content -- do not use this
                # branch if you need a clean generalization metric.
                image_range = range(image_length)
                if split == "test":
                    image_range = [image_range[0], image_range[int(image_length/3)], image_range[int(image_length*2/3)]]

            for i in image_range:
                num=i+1
                image_path=os.path.join(images_folder,f"frame_{str(num).zfill(5)}.{ext}")
                image_paths.append(image_path)
                image_poses.append((R,T))
                image_times.append(float(i/image_length))

        return image_paths, image_poses,image_times
    
    def get_video_cam_infos(self,datadir):
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        val_poses = get_spiral(poses, near_fars, N_views=N_views)

        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]
