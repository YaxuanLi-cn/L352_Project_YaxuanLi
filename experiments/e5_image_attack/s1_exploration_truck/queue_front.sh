#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
source /venv/rafel/bin/activate
until [ -f "$1" ]; do sleep 10; done
mkdir -p $STATE/locks_front
while read -r job; do
  lock="$STATE/locks_front/$(echo "$job" | tr ':' '_')"
  if mkdir "$lock" 2>/dev/null; then
    python run_ft.py --scene gsqa/truck --loss "$job" --init ctrl --steps 3000 --tag r1
  fi
done < jobs_front.txt
