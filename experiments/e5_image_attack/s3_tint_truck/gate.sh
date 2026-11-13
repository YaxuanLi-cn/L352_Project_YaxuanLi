#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
need=$1; est=$2; key=$3; shift 3
mkdir -p $STATE/gate
while :; do
  [ -f $STATE/STOP ] && { echo "$(date -u +%T) $key: STOP file, not started" >> $REC/logs/queue.log; exit 3; }
  if [ -f $REC/deadline.txt ] && [ $(( $(date +%s) + est )) -ge "$(cat $REC/deadline.txt)" ]; then
    echo "$(date -u +%T) $key: deadline too close, not started" >> $REC/logs/queue.log; exit 4
  fi
  if mkdir $STATE/gate/.lock 2>/dev/null; then
    used=0; for f in $STATE/gate/*.need; do [ -f "$f" ] && used=$((used + $(cat "$f"))); done
    if [ $((used + need)) -le 30000 ]; then echo "$need" > "$STATE/gate/$key.need"; rmdir $STATE/gate/.lock; break; fi
    rmdir $STATE/gate/.lock
  fi
  sleep 5
done
echo "$(date -u +%T) start $key (declared $need MiB)" >> $REC/logs/queue.log
"$@"; rc=$?
rm -f "$STATE/gate/$key.need"
echo "$(date -u +%T) end $key rc=$rc" >> $REC/logs/queue.log
exit $rc
