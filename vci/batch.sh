#!/usr/bin/env bash

# data 1 scaled
python prepare_data.py -i /data/schulz/bonn_tele_bench_001 -o ./data/multipleview/vci/b1_s4 -s4 &&
./multipleviewtria.sh vci/b1_s4 -i /data/schulz/bonn_tele_bench_001/calibration_dome.json -s4 &&
(
export DATA=multipleview/vci/b1_s4
export OUT=vci/b1_s4/v3_full
export ARG=multipleview/vci
export ITER="7000 14000 20000"
vci/train.sh &&
vci/render.sh &&
vci/metrics.sh
)

# data 2 scaled
python prepare_data.py -i /data/schulz/bonn_tele_bench_002 -o ./data/multipleview/vci/b2_s4 -s4 &&
./multipleviewtria.sh vci/b2_s4 -i /data/schulz/bonn_tele_bench_002/calibration_dome.json -s4 &&
(
export DATA=multipleview/vci/b2_s4
export OUT=vci/b2_s4/v3
export ARG=multipleview/vci
export ITER="7000 14000 20000"
vci/train.sh &&
vci/render.sh &&
vci/metrics.sh
)

# data 1 full resolution
python prepare_data.py -i /data/schulz/bonn_tele_bench_001 -o ./data/multipleview/vci/b1 &&
./multipleviewtria.sh vci/b1 -i /data/schulz/bonn_tele_bench_001/calibration_dome.json &&
(
export DATA=multipleview/vci/b1
export OUT=vci/b1/v3
export ARG=multipleview/vci
export ITER="7000 14000 20000"
vci/train.sh &&
vci/render.sh &&
vci/metrics.sh
)
