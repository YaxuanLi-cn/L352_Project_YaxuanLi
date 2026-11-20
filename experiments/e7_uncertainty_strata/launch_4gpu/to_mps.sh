#!/bin/bash
pid=$1; g=$2; name=$3; first=$4
. "$(dirname "$0")/env.sh" || exit 1
L=$LOGS/$name.log
n0=$(grep -c -E " DONE | FAILED " $L)
while [ "$(grep -c -E " DONE | FAILED " $L)" -le "$n0" ] && kill -0 $pid 2>/dev/null; do sleep 5; done
kill $pid 2>/dev/null
sleep 3
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps/log
CUDA_VISIBLE_DEVICES=$g exec python $E4/rerun_finetune.py $E4/configs/rerun.yaml --first "$first" >> $L 2>&1
