#!/bin/bash
g=$1; shift
. "$(dirname "$0")/env.sh" || exit 1
export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 TILE_MEM_FRACTION=0.34
P=$E7/configs/pilot.yaml
R=$E4/configs/rerun.yaml
for spec in "$@"; do
  ( d=${spec//\//_}
    until [ "$(ls $RERUN/stimuli/$d/seeds/*/done 2>/dev/null | wc -l)" -ge 4 ]; do
      python $E7/render_seeds.py $P $R $spec; sleep 60
    done
    until [ -f $PILOT/strata/$d.parquet ]; do python $E7/tiles.py strata --scenes $spec; sleep 60; done ) &
done
until [ "$(ls $SEEDS/*/base_seed*.pt 2>/dev/null | wc -l)" -ge 32 ]; do sleep 30; done
until python $E7/tiles.py metrics --scenes "$@" --loop; do sleep 30; done
wait
for spec in "$@"; do
  until python $E7/tiles.py pixel --scenes $spec; do sleep 30; done
  until python $E7/tiles.py ladder --scenes $spec; do sleep 30; done
  echo "$(date '+%m-%d %H:%M:%S') SCENE PIPELINE FINISHED $spec"
  touch $PILOT/scene_done_${spec//\//_}
done
