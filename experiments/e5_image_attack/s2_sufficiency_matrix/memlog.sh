#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
while ls $STATE/locks/worker_*.pid >/dev/null 2>&1; do
  echo "$(date -u +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ') $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | tr '\n' ' ')" >> $REC/logs/gpu_mem.log
  sleep 10
done
