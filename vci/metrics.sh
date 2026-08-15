#!/bin/sh

for i in ${NAME:-baseline conf_only resid_only traj_only combined}; do
	if [ "$DRY" = 1 ]; then
		echo python metrics.py -m "output/${OUT:-vci/b1_s4}/$i"
	else
		python metrics.py -m "output/${OUT:-vci/b1_s4}/$i"
	fi
done
