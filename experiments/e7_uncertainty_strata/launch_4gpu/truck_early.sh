#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
until [ -f $RERUN/runs/mip360_bicycle/s0/cvvdp__mixed/done.json ]; do sleep 20; done
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 TILE_MEM_FRACTION=0.25
if pgrep -f "nvidia-cuda-mps-contro[l]" > /dev/null; then export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps/log; fi
echo "$(date '+%m-%d %H:%M:%S') truck: tile metrics of finished runs (loop) and ladder"
( until python $E7/tiles.py metrics --scenes gsqa/truck --loop; do sleep 20; done ) &
python $E7/tiles.py ladder --scenes gsqa/truck
wait
echo "$(date '+%m-%d %H:%M:%S') TRUCK EARLY FINISHED"
