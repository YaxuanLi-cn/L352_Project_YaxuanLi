#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
while kill -0 469266 2>/dev/null; do sleep 3; done
echo "$(date -u +%H:%M:%S) flowers flip 0.4 finished; computing missing metric tables" >> $REC/logs/queue.log
/venv/rafel/bin/python ext_analysis.py --compute > $REC/logs/compute.log 2>&1
echo "$(date -u +%H:%M:%S) metric tables done rc=$?" >> $REC/logs/queue.log
