#!/bin/bash
pid=$1; g=$2; name=$3; mps=$4; shift 4
. "$(dirname "$0")/env.sh" || exit 1
L=$LOGS/$name.log
n0=$(grep -c -E " DONE | FAILED " $L)
while [ "$(grep -c -E " DONE | FAILED " $L)" -le "$n0" ] && kill -0 $pid 2>/dev/null; do sleep 3; done
kill $pid 2>/dev/null
sleep 3
if [ "$g" = "0" ]; then cpus="0-95,192-287"; else cpus="96-191,288-383"; fi
if [ "$mps" = "1" ]; then export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps/log; fi
CUDA_VISIBLE_DEVICES=$g exec taskset -c $cpus python $E4/rerun_finetune.py $E4/configs/rerun.yaml "$@" >> $L 2>&1
