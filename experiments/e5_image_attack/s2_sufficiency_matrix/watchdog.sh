#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
while [ "$(date +%s)" -lt "$(cat $REC/deadline.txt)" ]; do
  ls $STATE/locks/worker_*.pid >/dev/null 2>&1 || [ ! -f $STATE/locks/started ] || { echo "$(date -u +%T) watchdog: queue finished before the deadline" >> $REC/logs/queue.log; exit 0; }
  sleep 20
done
touch $STATE/STOP
for f in $STATE/locks/worker_*.pid $STATE/locks/job_*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done
echo "$(date -u +%T) watchdog: deadline reached, everything stopped" >> $REC/logs/queue.log
