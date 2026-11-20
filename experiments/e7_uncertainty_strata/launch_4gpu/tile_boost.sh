#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
until [ "$(ls $RERUN/runs/*/s*/*/done.json 2>/dev/null | wc -l)" -ge 180 ]; do sleep 30; done
echo "$(date '+%m-%d %H:%M:%S') all runs finished -> extra tile workers"
sleep 20
for g in 0 1 2 3; do
  for k in 1 2; do
    ( export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 TILE_MEM_FRACTION=0.3
      if pgrep -f "nvidia-cuda-mps-contro[l]" > /dev/null; then export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps/log; fi
      python $E7/tiles.py metrics >> $LOGS/tile_boost_g${g}_$k.log 2>&1 ) &
    sleep 3
  done
done
wait
echo "$(date '+%m-%d %H:%M:%S') TILE BOOST FINISHED"
