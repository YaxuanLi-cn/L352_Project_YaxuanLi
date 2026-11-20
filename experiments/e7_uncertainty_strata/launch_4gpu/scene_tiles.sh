#!/bin/bash
spec=$1
. "$(dirname "$0")/env.sh" || exit 1
export CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=8 TILE_MEM_FRACTION=0.3
if pgrep -f "nvidia-cuda-mps-contro[l]" > /dev/null; then export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps/log; fi
d=${spec//\//_}
until [ -f $PILOT/strata/$d.parquet ]; do python $E7/tiles.py strata --scenes $spec; sleep 20; done
n=3; pgrep -f "rerun_finetun[e].py" > /dev/null && n=1
for k in $(seq $n); do ( until python $E7/tiles.py metrics --scenes $spec; do sleep 20; done ) & sleep 5; done
wait
until python $E7/tiles.py metrics --scenes $spec; do sleep 20; done
if pgrep -f "rerun_finetun[e].py" > /dev/null; then
  until python $E7/tiles.py pixel --scenes $spec; do sleep 20; done
  until python $E7/tiles.py ladder --scenes $spec; do sleep 20; done
else
  ( until python $E7/tiles.py pixel --scenes $spec; do sleep 20; done ) &
  ( until python $E7/tiles.py ladder --scenes $spec; do sleep 20; done ) &
  wait
fi
echo "$(date '+%m-%d %H:%M:%S') SCENE PIPELINE FINISHED $spec"
touch $PILOT/scene_done_$d
