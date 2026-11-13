#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
CAP=30000
used() { local u=0; for f in $STATE/gate/*.need; do [ -f "$f" ] && u=$((u + $(cat "$f"))); done; echo "$u"; }
while read -r scene need est; do
  [ -z "$scene" ] && continue
  case "$scene" in \#*) continue;; esac
  if [ -f $STATE/STOP ] || [ $(( $(date +%s) + est )) -ge "$(cat $REC/gpu_deadline.txt)" ]; then
    echo "$(date -u +%T) cut: $scene and every scene after it (three levels would not finish before the deadline)" >> $REC/logs/queue.log
    break
  fi
  for mult in 1 0.5 2; do
    until [ $(( $(used) + need )) -le $CAP ]; do sleep 3; [ -f $STATE/STOP ] && break 3; done
    key="${scene//\//_}__x$mult"
    echo "$need" > "$STATE/gate/$key.need"
    echo "$(date -u +%T) start $key (declared $need MiB)" >> $REC/logs/queue.log
    ( /venv/rafel/bin/python run_colour.py --scene "$scene" --mult "$mult" > "$REC/logs/$key.log" 2>&1; rc=$?; rm -f "$STATE/gate/$key.need"; echo "$(date -u +%T) end $key rc=$rc" >> $REC/logs/queue.log ) &
    sleep 3
  done
done < scenes.txt
wait
echo "$(date -u +%T) dispatcher finished" >> $REC/logs/queue.log
