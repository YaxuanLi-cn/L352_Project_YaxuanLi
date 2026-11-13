#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
W=$1
JOBS=${2:-jobs.txt}
echo $$ > $STATE/locks/worker_$W.pid
while read -r grp scene loss seed; do
  [ -z "$grp" ] && continue
  case "$grp" in \#*) continue;; esac
  [ -f $STATE/STOP ] && break
  [ -f "$STATE/STOP_$W" ] && break
  key="${scene//\//_}__${loss//:/-}__s${seed}"
  [ -d "$STATE/locks/$key" ] && continue
  est=$(awk -v s="$scene" '$1==s{print $2}' $REC/est_seconds.txt)
  now=$(date +%s)
  if [ $((now + est)) -ge "$(cat $REC/deadline.txt)" ]; then
    echo "$(date -u +%T) worker $W: deadline too close, not starting $key (and nothing after it)" >> $REC/logs/queue.log
    break
  fi
  mkdir "$STATE/locks/$key" 2>/dev/null || continue
  echo "$(date -u +%T) worker $W: start $grp $key" >> $REC/logs/queue.log
  /venv/rafel/bin/python run_suff.py --scene "$scene" --loss "$loss" --seed "$seed" > "$REC/logs/$key.log" 2>&1 &
  echo $! > $STATE/locks/job_$W.pid
  wait $!
  echo "$(date -u +%T) worker $W: end $grp $key rc=$?" >> $REC/logs/queue.log
done < "$JOBS"
echo "$(date -u +%T) worker $W: finished" >> $REC/logs/queue.log
rm -f $STATE/locks/worker_$W.pid $STATE/locks/job_$W.pid
