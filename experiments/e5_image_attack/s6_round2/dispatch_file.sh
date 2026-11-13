#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
run_one() {
  scene=$1; target=$2; gamma=$3; w=$4; extra=$5
  tag=$(echo "${scene}_${target}_${gamma}_w${w}${extra:+_h$extra}" | tr '/' '_')
  echo "$(date -u +%H:%M:%S) start $tag" >> $REC/logs/queue.log
  /venv/rafel/bin/python run_ext.py --scene "$scene" --target "$target" --gamma "$gamma" --wdefault "$w" ${extra:+--hold-extra "$extra"} > "$REC/logs/$tag.log" 2>&1
  rc=$?
  echo "$(date -u +%H:%M:%S) end $tag rc=$rc" >> $REC/logs/queue.log
}
export -f run_one
xargs -P "$2" -L 1 bash -c 'run_one "$@"' _ < "$1"
echo "$(date -u +%H:%M:%S) $1 finished" >> $REC/logs/queue.log
touch "$REC/logs/done_$(basename "$1")"
