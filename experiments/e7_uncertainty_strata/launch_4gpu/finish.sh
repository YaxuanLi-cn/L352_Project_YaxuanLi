#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
export CUDA_VISIBLE_DEVICES=${1:-0} OMP_NUM_THREADS=16
until [ "$(ls $PILOT/scene_done_* 2>/dev/null | wc -l)" -ge 8 ]; do sleep 60; done
set -x
python $E7/tiles.py strata
python $E7/tiles.py checks
python $E7/tiles.py table
python $E7/analysis.py
python $E7/figures.py
python $E4/recompute_whole.py
python $E4/compare_whole.py
echo "$(date '+%m-%d %H:%M:%S') FINISH DONE"
