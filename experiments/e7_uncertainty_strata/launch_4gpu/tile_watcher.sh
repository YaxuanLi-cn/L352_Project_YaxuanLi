#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
pending="gsqa/flowers gsqa/train gsqa/playground2 mip360/kitchen mip360/bonsai mip360/garden mip360/bicycle gsqa/truck"
while [ -n "$pending" ]; do
  next=""
  for spec in $pending; do
    d=${spec//\//_}; need=18; [ "$spec" = "gsqa/truck" ] && need=54
    n=$(ls $RERUN/runs/$d/s*/*/done.json 2>/dev/null | wc -l)
    if [ "$n" -ge "$need" ]; then
      read g free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',' | sort -k2 -n -r | head -1)
      if [ "$free" -ge 10000 ]; then
        echo "$(date '+%m-%d %H:%M:%S') $spec: all $n runs finished -> tile steps on GPU $g ($free MiB free)"
        setsid nohup $E7/launch_4gpu/scene_tiles.sh $spec $g >> $LOGS/tiles_$d.log 2>&1 < /dev/null &
        sleep 60
        continue
      fi
    fi
    next="$next $spec"
  done
  pending=$next
  sleep 60
done
echo "$(date '+%m-%d %H:%M:%S') TILE WATCHER FINISHED"
