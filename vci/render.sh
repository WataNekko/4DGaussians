#!/bin/sh

for i in ${NAME:-baseline conf_only resid_only traj_only combined}; do
	for iter in ${ITER:-14000 20000}; do
		if [ "$DRY" = 1 ]; then
			echo python render.py -m "output/${OUT:-vci/b1_s4}/$i" --skip_train --skip_video --skip_custom --iteration $iter
		else
			python render.py -m "output/${OUT:-vci/b1_s4}/$i" --skip_train --skip_video --skip_custom --iteration $iter
		fi
	done
done
