#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
scene=$1; need=$2; est=$3; shift 3
sdn=${scene//\//_}
for g in "$@"; do
  bash gate.sh "$need" "$est" "${sdn}__$g" /venv/rafel/bin/python run_tint.py --scene "$scene" --loss "front:triplet:dreamsim:$g" > "$REC/logs/${sdn}__$g.log" 2>&1 &
done
wait
g2=$(/venv/rafel/bin/python ladder.py next "$scene" 2>> "$REC/logs/next_$sdn.err" | tail -n 1)
echo "$(date -u +%T) $scene: round-2 level = $g2" >> $REC/logs/queue.log
if [ -z "$g2" ] || [ "$g2" = "none" ]; then exit 0; fi
for f in $WAIT_FOR; do until [ -f "$f" ] || [ -f $STATE/STOP ]; do sleep 5; done; done
touch "$REC/logs/r2_requested_$sdn"
bash gate.sh "$need" "$est" "${sdn}__$g2" /venv/rafel/bin/python run_tint.py --scene "$scene" --loss "front:triplet:dreamsim:$g2" > "$REC/logs/${sdn}__$g2.log" 2>&1
echo "$(date -u +%T) $scene: chain finished" >> $REC/logs/queue.log
