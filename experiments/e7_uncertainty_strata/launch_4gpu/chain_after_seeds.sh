#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "$(date '+%m-%d %H:%M:%S') seed worker $1 finished; starting the stage-B re-run (worker $2)"
if [ "$2" = "2" ]; then R="--reverse"; else R=""; fi
python $E4/rerun_finetune.py $E4/configs/rerun.yaml $R
if [ "$2" = "1" ]; then
  python $E7/train_seeds.py $E7/configs/pilot.yaml
  python $E7/render_seeds.py $E7/configs/pilot.yaml $E4/configs/rerun.yaml
fi
echo "$(date '+%m-%d %H:%M:%S') CHAIN $2 FINISHED"
