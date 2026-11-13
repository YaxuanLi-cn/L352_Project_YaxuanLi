#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
grp=$1
while :; do
  all=1; n=0; k=0
  while read -r g scene loss seed; do
    [ "$g" = "$grp" ] || continue
    n=$((n + 1))
    if [ -f "$RUNS/${scene//\//_}/${loss//:/-}__from-ctrl__3000__s${seed}/done.json" ]; then k=$((k + 1)); else all=0; fi
  done < jobs.txt
  [ $all = 1 ] && { echo "group $grp complete ($k/$n)"; break; }
  ls $STATE/locks/worker_*.pid >/dev/null 2>&1 || { echo "group $grp: no worker left ($k/$n done)"; break; }
  sleep 15
done
date -u; tail -6 $REC/logs/queue.log
