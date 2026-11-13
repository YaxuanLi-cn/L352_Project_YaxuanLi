#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
run_one() {
  scene=$1; target=$2; gamma=$3
  tag=$(echo "${scene}_${target}_${gamma}" | tr '/' '_')
  if [ "$(date +%s)" -gt "$(cat $REC/start_deadline.txt)" ]; then echo "$(date -u +%H:%M:%S) skip $tag (start deadline passed)" >> $REC/logs/queue.log; return; fi
  echo "$(date -u +%H:%M:%S) start $tag" >> $REC/logs/queue.log
  /venv/rafel/bin/python run_ext.py --scene "$scene" --target "$target" --gamma "$gamma" > "$REC/logs/$tag.log" 2>&1
  echo "$(date -u +%H:%M:%S) end $tag rc=$?" >> $REC/logs/queue.log
}
export -f run_one
xargs -P 4 -L 1 bash -c 'run_one "$@"' _ < jobs.txt
echo "$(date -u +%H:%M:%S) dispatcher finished" >> $REC/logs/queue.log
touch $REC/logs/all_done
