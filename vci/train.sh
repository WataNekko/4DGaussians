#!/usr/bin/env bash

for i in ${NAME:-baseline conf_only resid_only traj_only combined}; do
	EXPNAME="${OUT:-vci/b1_s4}/$i"
	OUT_DIR="output/$EXPNAME"
	TRAIN_LOG="$OUT_DIR/train.log"
	if [ "$DRY" = 1 ]; then
		echo python train.py -s "data/${DATA:-multipleview/b1_s4_tria/}" --expname "$EXPNAME" --configs "arguments/${ARG:-multipleview/vci}/$i.py"
	else
		mkdir -p "$OUT_DIR"
		script -q -e -a -c "python train.py -s \"data/${DATA:-multipleview/b1_s4_tria/}\" --expname \"$EXPNAME\" --configs \"arguments/${ARG:-multipleview/vci}/$i.py\"" "$TRAIN_LOG"
	fi
done
