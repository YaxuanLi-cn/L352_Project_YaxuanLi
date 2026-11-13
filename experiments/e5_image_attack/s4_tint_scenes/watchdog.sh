#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
while [ "$(date +%s)" -lt "$(cat $REC/gpu_deadline.txt)" ]; do
  [ -f $REC/logs/all_done ] && exit 0
  sleep 15
done
touch $STATE/STOP
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill "$p" 2>/dev/null; done
echo "$(date -u +%T) watchdog: GPU deadline reached, running jobs killed" >> $REC/logs/queue.log
