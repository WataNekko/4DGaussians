#!/usr/bin/env bash

workdir=$1

python convert_calib.py "${@:2}" -o ./colmap_tmp/txt -r || exit 2

mkdir -p ./colmap_tmp/sparse/0
colmap model_converter --input_path ./colmap_tmp/txt --output_path ./colmap_tmp/sparse/0 --output_type BIN

./scripts/clone_llff.sh
python LLFF/imgs2poses.py ./colmap_tmp/

mv ./colmap_tmp/poses_bounds.npy ./data/multipleview/$workdir/poses_bounds_multipleview.npy
mv ./colmap_tmp/sparse/0/points3D.bin ./data/multipleview/$workdir/points3D_multipleview.bin

mkdir ./data/multipleview/$workdir/sparse_
mv ./colmap_tmp/sparse/0/*.bin ./data/multipleview/$workdir/sparse_

rm -rf ./colmap_tmp
rm -rf ./LLFF
