#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
run_one() {
  scene=$1; target=$2; gamma=$3
  tag=$(echo "${scene}_${target}_${gamma}" | tr '/' '_')
  echo "$(date -u +%H:%M:%S) start $tag" >> $REC/logs/queue.log
  /venv/rafel/bin/python run_ext.py --scene "$scene" --target "$target" --gamma "$gamma" > "$REC/logs/$tag.log" 2>&1
  echo "$(date -u +%H:%M:%S) end $tag rc=$?" >> $REC/logs/queue.log
}
export -f run_one
printf 'gsqa/truck multi 0.05\ngsqa/train multi 0.05\ngsqa/playground2 multi 0.05\n' | xargs -P 3 -L 1 bash -c 'run_one "$@"' _
while kill -0 453492 2>/dev/null; do sleep 3; done
echo "$(date -u +%H:%M:%S) end gsqa_flowers_multi_0.8 (started by the first dispatcher)" >> $REC/logs/queue.log
printf 'gsqa/truck flip 0.4\ngsqa/flowers flip 0.4\ngsqa/playground2 flip 0.4\ngsqa/train flip 0.4\n' | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo "$(date -u +%H:%M:%S) dispatcher 3 finished" >> $REC/logs/queue.log
touch $REC/logs/all_done3
