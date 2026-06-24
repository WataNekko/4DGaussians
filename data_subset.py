import os
import shutil

# Use absolute paths to prevent broken symlinks
src_dir = os.path.abspath("data/multipleview/bonn_tele_bench_001")
subset_dir = os.path.abspath("data/multipleview/bonn_tele_bench_001_subset")

max_frames = 12  # Keep this low (30-50) for a quick memory-safe test

os.makedirs(subset_dir, exist_ok=True)

# 1. Copy the sparse directory (camera poses and pointcloud)
# This is crucial so you don't have to regenerate the binaries
for sparse_name in ["sparse", "sparse_"]: # Catch both common naming conventions
    src_sparse = os.path.join(src_dir, sparse_name)
    if os.path.exists(src_sparse):
        shutil.copytree(src_sparse, os.path.join(subset_dir, sparse_name), dirs_exist_ok=True)
        print(f"Copied {sparse_name} directory.")

# 2. Symlink the camera folders, limiting the number of frames
cam_folders = sorted([d for d in os.listdir(src_dir) if d.startswith("cam")])

for cam in cam_folders:
    src_cam_dir = os.path.join(src_dir, cam)
    target_cam_dir = os.path.join(subset_dir, cam)
    if os.path.exists(target_cam_dir):
        shutil.rmtree(target_cam_dir)
    os.makedirs(target_cam_dir)
    
    frames = sorted([f for f in os.listdir(src_cam_dir) if f.endswith(".jpg")])
    
    for i, frame in enumerate(frames[20:20+max_frames]):
        src_frame = os.path.join(src_cam_dir, frame)
        target_frame = os.path.join(target_cam_dir, f"frame_{i + 1:05d}.jpg")
        if not os.path.exists(target_frame):
            os.symlink(src_frame, target_frame)

print(f"Success! Created subset with {len(cam_folders)} cameras and {min(max_frames, len(frames))} frames per camera.")
