#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
until [ ! -f $STATE/locks/worker_5.pid ] && [ -f "$RUNS/mip360_garden/front-triplet-flip-0.05__from-ctrl__3000__s0/done.json" ]; do
  [ -f $STATE/STOP ] && exit 0
  sleep 5
done
rmdir $STATE/locks/mip360_garden__front-triplet-dreamsim-0.0125__s0
echo "$(date -u +%T) relay: starting worker 6 for X garden" >> $REC/logs/queue.log
exec bash queue3.sh 6 jobs_xgarden.txt > $REC/logs/worker_6.out 2>&1
