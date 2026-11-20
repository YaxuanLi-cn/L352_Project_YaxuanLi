#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
date '+%m-%d %H:%M:%S UTC'
echo "seed models: $(ls $SEEDS/*/base_seed*.pt | wc -l)/32   seed renders: $(ls $RERUN/stimuli/*/seeds/*/done 2>/dev/null | wc -l)/32   strata: $(ls $PILOT/strata/*.parquet 2>/dev/null | wc -l)/8"
tot=0
for d in gsqa_truck/s0 gsqa_truck/s1 gsqa_truck/s2 gsqa_train/s0 gsqa_playground2/s0 gsqa_flowers/s0 mip360_kitchen/s0 mip360_bonsai/s0 mip360_garden/s0 mip360_bicycle/s0; do
  n=$(ls $RERUN/runs/$d/*/done.json 2>/dev/null | wc -l); tot=$((tot+n))
  sc=${d%/*}; sd=${d#*/}
  t=$(ls $PILOT/cache/$sc/test/$sd/*.parquet 2>/dev/null | wc -l); r=$(ls $PILOT/cache/$sc/train/$sd/*.parquet 2>/dev/null | wc -l)
  echo "  $d: runs $n/18   tile metrics test $t/17 train $r/17"
done
need=0
for d in gsqa_truck gsqa_train gsqa_playground2 gsqa_flowers mip360_kitchen mip360_bonsai mip360_garden mip360_bicycle; do
  for m in default__ctrl msssim__pure lpips_vgg__pure lpips_alex__pure dists__pure cvvdp__pure lpips_vgg__mixed lpips_alex__mixed dists__mixed flip__mixed; do
    [ -f $RERUN/runs/$d/s0/$m/done.json ] && need=$((need+1))
  done
done
echo "runs done: $tot/180   runs needed for the study stimuli: $need/80   scene pipelines finished: $(ls $PILOT/scene_done_* 2>/dev/null | wc -l)/8"
echo "workers: rerun $(pgrep -f 'rerun_finetun[e].py' | wc -l), seeds $(pgrep -f 'train_seed[s].py' | wc -l), tile $(pgrep -f 'tile[s].py' | wc -l)"
grep -l -E "Traceback" $LOGS/rerun_g*.log $LOGS/pipe_*.log $LOGS/seeds_g*.log $LOGS/finish.log 2>/dev/null | tr '\n' ' '
echo; df -h / | tail -1 | awk '{print "disk free: "$4}'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; echo
