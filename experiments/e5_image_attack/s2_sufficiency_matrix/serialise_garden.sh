#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
key=mip360_garden__front-default-dists-0.05__s0
until [ -d "$STATE/locks/$key" ]; do sleep 3; done
sleep 2
w=$(grep "start C $key" $REC/logs/queue.log | tail -n 1 | sed -e 's/.*worker \([0-9]*\):.*/\1/')
touch "$STATE/STOP_$w"
echo "$(date -u +%T) serialise: worker $w will stop after $key" >> $REC/logs/queue.log
while [ -f "$STATE/locks/worker_$w.pid" ]; do sleep 3; done
rm -f "$STATE/STOP_$w"
echo "$(date -u +%T) serialise: worker $w gone, starting worker 5 on jobs_small.txt" >> $REC/logs/queue.log
exec bash queue3.sh 5 jobs_small.txt > $REC/logs/worker_5.out 2>&1
