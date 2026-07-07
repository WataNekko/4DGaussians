#!/usr/bin/env bash

workdir=$1

# 1. Extract and match features normally
python scripts/extractimages.py multipleview/$workdir
colmap feature_extractor --database_path ./colmap_tmp/database.db --image_path ./colmap_tmp/images --ImageReader.camera_model OPENCV --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
colmap exhaustive_matcher --database_path ./colmap_tmp/database.db

# 2. Convert calibration params files
python convert_calib.py "${@:2}" -o ./colmap_tmp/sparse_input --db ./colmap_tmp/database.db || exit 2

# 3. Triangulate using KNOWN poses
mkdir ./colmap_tmp/sparse_output
colmap point_triangulator --database_path ./colmap_tmp/database.db --image_path ./colmap_tmp/images --input_path ./colmap_tmp/sparse_input --output_path ./colmap_tmp/sparse_output

# 4. Save the populated sparse model for 4DGS to use
mkdir ./data/multipleview/$workdir/sparse_
cp -r ./colmap_tmp/sparse_output/* ./data/multipleview/$workdir/sparse_

# 5. Continue to Dense Reconstruction...
mkdir ./colmap_tmp/dense
colmap image_undistorter --image_path ./colmap_tmp/images --input_path ./colmap_tmp/sparse_output --output_path ./colmap_tmp/dense --output_type COLMAP
colmap patch_match_stereo --workspace_path ./colmap_tmp/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path ./colmap_tmp/dense --workspace_format COLMAP --input_type geometric --output_path ./colmap_tmp/dense/fused.ply

python scripts/downsample_point.py ./colmap_tmp/dense/fused.ply ./data/multipleview/$workdir/points3D_multipleview.ply

./scripts/clone_llff.sh
python LLFF/imgs2poses.py ./colmap_tmp/

cp ./colmap_tmp/poses_bounds.npy ./data/multipleview/$workdir/poses_bounds_multipleview.npy

rm -rf ./colmap_tmp
rm -rf ./LLFF
