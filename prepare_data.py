import os
import argparse
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_single_image(args):
    src_rgb, src_mask, target_path, scale, apply_mask = args

    try:
        # Load as RGBA if masking, otherwise standard RGB
        if apply_mask and os.path.exists(src_mask):
            img = Image.open(src_rgb).convert("RGBA")
        else:
            img = Image.open(src_rgb).convert("RGB")

        # Scale
        if scale > 1:
            new_size = (int(img.width / scale), int(img.height / scale))
            img = img.resize(new_size, Image.LANCZOS)

        img_np = np.array(img)

        # Mask - Pre-multiply RGB and inject Alpha
        if apply_mask and os.path.exists(src_mask):
            mask = Image.open(src_mask).convert("L")
            if scale > 1:
                mask = mask.resize(new_size, Image.NEAREST)

            mask_normalized = np.array(mask) / 255.0

            # 1. Multiply the RGB channels (indices 0, 1, 2) by the mask
            img_np[..., :3] = (img_np[..., :3] * mask_normalized[..., None]).astype(np.uint8)

            # 2. Overwrite the 4th channel (Alpha) with the grayscale mask
            img_np[..., 3] = np.array(mask)

        # Save
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        Image.fromarray(img_np).save(target_path)
        return True
    except Exception as e:
        return str(e)

def main():
    parser = argparse.ArgumentParser(description="Prepare 4DGS Dome Data")
    parser.add_argument("-i", "--source", type=str, required=True, help="Raw data directory")
    parser.add_argument("-o", "--target", type=str, required=True, help="Output directory")
    parser.add_argument("-s", "--scale", type=float, default=1, help="Downscale factor (e.g. 2 for half size)")
    parser.add_argument("-m", "--mask", action="store_true", help="Apply masks to RGB images")
    parser.add_argument("-f", "--from", type=int, default=1, help="The frame to start from")
    parser.add_argument("-t", "--to", type=int, help="The frame to end with")
    args = parser.parse_args()

    # 1. Scan frames and explicitly ignore the background frame (frame_00000)
    frames = sorted([f for f in os.listdir(args.source) if f.startswith("frame_") and f != "frame_00000"])

    if not frames:
        print("No valid frames found after skipping frame_00000.")
        return

    from_ = getattr(args, "from") - 1
    to = args.to if args.to is not None else len(frames)
    frames = frames[from_:to]

    # 2. Build the continuous camera index mapping (cam01, cam02, etc.)
    # We scan the first valid frame to establish the baseline camera roster
    first_frame_rgb_dir = os.path.join(args.source, frames[0], "rgb")
    if not os.path.exists(first_frame_rgb_dir):
        print(f"Error: Could not find rgb directory in {first_frame_rgb_dir}")
        return

    raw_cam_images = [img for img in os.listdir(first_frame_rgb_dir) if img.endswith(".jpg")]

    # Extract the original IDs ('C0000', 'C0001', 'C0003') and sort them
    raw_cam_ids = sorted([img.split('.')[0] for img in raw_cam_images])

    # Create the mapping dictionary: e.g., C0000 -> 01, C0003 -> 03
    cam_mapping = {raw_id: f"{idx + 1:02d}" for idx, raw_id in enumerate(raw_cam_ids)}

    print(f"Established continuous camera mapping for {len(cam_mapping)} physical cameras.")

    # 3. Build task list
    tasks = []
    print(f"Scanning {len(frames)} active frames ({from_+1}..={to})...")

    for frame_idx, frame_name in enumerate(frames):
        rgb_dir = os.path.join(args.source, frame_name, "rgb")
        mask_dir = os.path.join(args.source, frame_name, "mask")

        if not os.path.exists(rgb_dir):
            continue

        images = [img for img in os.listdir(rgb_dir) if img.endswith(".jpg")]

        for img_name in images:
            cam_id = img_name.split('.')[0]

            # Failsafe: Ignore random cameras that weren't in the initial roster mapping
            if cam_id not in cam_mapping:
                continue

            mapped_cam_id = cam_mapping[cam_id]

            src_rgb = os.path.join(rgb_dir, img_name)
            src_mask = os.path.join(mask_dir, f"mask_{img_name}")

            # Apply the mapping to the output directory (e.g., cam_01, cam_02)
            target_cam_dir = os.path.join(args.target, f"cam{mapped_cam_id}")

            # NEW: Switch to PNG if we are generating transparency
            ext = ".png" if args.mask else ".jpg"

            # Frame indexing remains 1-indexed, starting at frame_00001
            target_path = os.path.join(target_cam_dir, f"frame_{frame_idx + 1:05d}{ext}")

            tasks.append((src_rgb, src_mask, target_path, args.scale, args.mask))

    print(f"Processing {len(tasks)} images with {os.cpu_count()} workers...")

    # 4. Execute in parallel
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))

    failures = [r for r in results if r is not True]
    if failures:
        print(f"Finished with {len(failures)} errors. First error: {failures[0]}")
    else:
        print(f"Success! Data saved to {args.target}")

if __name__ == "__main__":
    main()
