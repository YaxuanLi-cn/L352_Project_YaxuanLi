#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
P=$E7/configs/pilot.yaml
R=$E4/configs/rerun.yaml
L=$LOGS
A=("gsqa/truck s0|gsqa/truck s1|mip360/bicycle|mip360/garden"
   "gsqa/truck s0|gsqa/truck s2|mip360/garden|mip360/bicycle"
   "gsqa/truck s0|mip360/kitchen|mip360/bicycle|mip360/garden"
   "mip360/bonsai|mip360/garden|mip360/bicycle")
B=("mip360/bicycle|mip360/garden"
   "mip360/garden|mip360/bicycle"
   "gsqa/train|gsqa/flowers|mip360/bicycle|mip360/garden"
   "gsqa/playground2|mip360/garden|mip360/bicycle")
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g setsid nohup python $E4/rerun_finetune.py $R --first "${A[$g]}" \
      >> $L/rerun_g${g}a.log 2>&1 < /dev/null &
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "python $E7/train_seeds.py $P >> $L/seeds_g${g}.log 2>&1; \
      python $E4/rerun_finetune.py $R --first '${B[$g]}' >> $L/rerun_g${g}b.log 2>&1" \
      > /dev/null 2>&1 < /dev/null &
  sleep 2
done
