#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
run_one() {
  scene=$1; target=$2; gamma=$3; w=$4
  tag=$(echo "${scene}_${target}_${gamma}_w${w}" | tr '/' '_')
  echo "$(date -u +%H:%M:%S) start $tag" >> $REC/logs/queue.log
  /venv/rafel/bin/python run_ext.py --scene "$scene" --target "$target" --gamma "$gamma" --wdefault "$w" > "$REC/logs/$tag.log" 2>&1
  echo "$(date -u +%H:%M:%S) end $tag rc=$?" >> $REC/logs/queue.log
}
export -f run_one
printf 'gsqa/truck multi 0.2 3\ngsqa/train multi 0.2 3\ngsqa/playground2 multi 0.2 3\n' | xargs -P 3 -L 1 bash -c 'run_one "$@"' _
echo "$(date -u +%H:%M:%S) dispatcher 4 finished" >> $REC/logs/queue.log
touch $REC/logs/all_done4
