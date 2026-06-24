import os

# Update these to point to your actual directories
source_dir = os.path.abspath("/data/schulz/bonn_tele_bench_001")
target_dir = os.path.abspath("./data/multipleview/bonn_tele_bench_001")

# Get all frame directories, ensuring chronological order
frames = sorted([f for f in os.listdir(source_dir) if f.startswith("frame_")])

link_count = 0

cam_map = None

for frame_idx, frame_name in enumerate(frames):
    rgb_dir = os.path.join(source_dir, frame_name, "rgb")
    
    # Skip if the rgb directory doesn't exist for some reason
    if not os.path.exists(rgb_dir):
        continue
        
    images = [img for img in os.listdir(rgb_dir) if img.endswith(".jpg")]

    if cam_map is None:
        cam_map = {img: f"cam{i+1:02d}" for i, img in enumerate(sorted(images))}
    
    for img_name in images:
        # Create the target camera directory (e.g., cam01)
        target_cam_dir = os.path.join(target_dir, cam_map[img_name])
        os.makedirs(target_cam_dir, exist_ok=True)
        
        # Define source and target file paths
        src_file = os.path.join(rgb_dir, img_name)
        
        # 4DGS multipleview expects 1-indexed, zero-padded frames (e.g., frame_00001.jpg)
        target_file = os.path.join(target_cam_dir, f"frame_{frame_idx + 1:05d}.jpg")
        
        # Create the symlink if it doesn't already exist
        if not os.path.exists(target_file):
            os.symlink(src_file, target_file)
            link_count += 1

print(f"Success! Created {link_count} symlinks in {target_dir}")
