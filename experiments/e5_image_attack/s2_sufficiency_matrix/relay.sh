#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
while [ -f "$STATE/locks/worker_$1.pid" ]; do sleep 3; done
rm -f "$STATE/STOP_$1"
echo "$(date -u +%T) relay: worker $1 gone, starting worker $2" >> $REC/logs/queue.log
exec bash queue2.sh "$2" > "$REC/logs/worker_$2.out" 2>&1
