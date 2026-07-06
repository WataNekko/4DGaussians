import os
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def main():
    parser = argparse.ArgumentParser(description="Convert Dome Calibration JSON to COLMAP format.")
    parser.add_argument("-i", "--json", type=str, required=True, help="Path to calibration_dome.json")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output sparse/0/ directory")
    parser.add_argument("-s", "--scale", type=int, default=1, help="Downscale factor (e.g., 2, 4, 8)")
    parser.add_argument("-r", "--randomize_points", action="store_true", help="Generate random points. Omit to leave empty for COLMAP triangulation.")
    parser.add_argument("-n", "--num_points", type=int, default=100000, help="Number of random points if --randomize_points is set.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.json, 'r') as f:
        calib = json.load(f)

    # Sort cameras alphabetically by camera_id (C0000, C0001, etc.) to ensure strict 1-indexed routing
    cameras_list = sorted(calib['cameras'], key=lambda x: x['camera_id'])

    cameras_txt = open(os.path.join(args.output, "cameras.txt"), "w")
    images_txt = open(os.path.join(args.output, "images.txt"), "w")
    points_txt = open(os.path.join(args.output, "points3D.txt"), "w")

    cameras_txt.write("# Camera list with one line of data per camera:\n")
    cameras_txt.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    
    images_txt.write("# Image list with two lines of data per image:\n")
    images_txt.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    images_txt.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
    
    points_txt.write("# 3D point list with one line of data per point:\n")
    points_txt.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

    # The JSON values are raw Bayer.
    # We must first scale by 2 to reach the RGB baseline, then apply the user's RGB downscale factor.
    bayer_factor = 2.0
    total_scale = bayer_factor * args.scale

    for cam_idx, cam_data in enumerate(cameras_list, start=1):
        # 1. Intrinsics Extraction (Scaled for Debayering + RGB Downscaling)
        w = int(cam_data['intrinsics']['resolution'][0] / total_scale)
        h = int(cam_data['intrinsics']['resolution'][1] / total_scale)

        cam_matrix = cam_data['intrinsics']['camera_matrix']
        fx = cam_matrix[0] / total_scale
        fy = cam_matrix[4] / total_scale
        cx = cam_matrix[2] / total_scale
        cy = cam_matrix[5] / total_scale

        # Grab the first 4 distortion coefficients (k1, k2, p1, p2)
        dist = cam_data['intrinsics']['distortion_coefficients']
        k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]

        cameras_txt.write(f"{cam_idx} OPENCV {w} {h} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n")
        
        # 2. Extrinsics Extraction
        # Reshape the flat 16-element view matrix into a 4x4 NumPy array
        view_mat = np.array(cam_data['extrinsics']['view_matrix']).reshape(4, 4)
        rot_mat = view_mat[:3, :3]
        t_vec = view_mat[:3, 3]
        
        # Convert Rotation Matrix to Quaternion (SciPy yields [qx, qy, qz, qw], COLMAP wants [qw, qx, qy, qz])
        quat = R.from_matrix(rot_mat).as_quat()
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        
        # Assumes multipleviewprogress.sh extracts frames as image1.jpg, image2.jpg...
        image_name = f"image{cam_idx}.jpg"
        
        images_txt.write(f"{cam_idx} {qw} {qx} {qy} {qz} {t_vec[0]} {t_vec[1]} {t_vec[2]} {cam_idx} {image_name}\n\n")

    # 3. Point Cloud Logic
    if args.randomize_points:
        print(f"Generating {args.num_points} random initialization points...")
        # Since C0004 is the origin, we shift the random points +2.5m along the Z-axis 
        # so they initialize in front of the camera lenses, not behind them.
        xyz = np.random.uniform(-2.0, 2.0, (args.num_points, 3))
        xyz[:, 2] += 2.5 
        colors = np.random.randint(0, 255, (args.num_points, 3))
        
        for i in range(args.num_points):
            points_txt.write(f"{i+1} {xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {colors[i,0]} {colors[i,1]} {colors[i,2]} 0.0\n")
        print("Random points populated.")
    else:
        print("Skipping point randomization. points3D.txt left empty for COLMAP triangulation.")

    cameras_txt.close()
    images_txt.close()
    points_txt.close()
    print(f"Success! Native COLMAP text files written to {args.output}")

if __name__ == "__main__":
    main()
