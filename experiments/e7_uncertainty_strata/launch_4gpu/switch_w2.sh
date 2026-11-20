#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
L=$LOGS/seeds_w2.log
n0=$(grep -c "DONE" $L)
while [ "$(grep -c "DONE" $L)" -le "$n0" ]; do sleep 10; done
kill "$1"
echo "$(date '+%m-%d %H:%M:%S') seed worker 2 stopped after its job; starting the stage-B re-run (forward)"
python $E4/rerun_finetune.py $E4/configs/rerun.yaml
echo "$(date '+%m-%d %H:%M:%S') RERUN WORKER 2 FINISHED"
