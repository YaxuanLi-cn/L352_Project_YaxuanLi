#!/bin/bash
spec=$1
. "$(dirname "$0")/env.sh" || exit 1
export CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=8
P=$E7/configs/pilot.yaml
R=$E4/configs/rerun.yaml
d=${spec//\//_}
until [ "$(ls $RERUN/stimuli/$d/seeds/*/done 2>/dev/null | wc -l)" -ge 4 ]; do
  python $E7/render_seeds.py $P $R $spec; sleep 60
done
until [ -f $PILOT/strata/$d.parquet ]; do python $E7/tiles.py strata --scenes $spec; sleep 60; done
until python $E7/tiles.py metrics --scenes $spec --loop; do sleep 30; done
until python $E7/tiles.py pixel --scenes $spec; do sleep 30; done
until python $E7/tiles.py ladder --scenes $spec; do sleep 30; done
echo "$(date '+%m-%d %H:%M:%S') SCENE PIPELINE FINISHED $spec"
touch $PILOT/scene_done_$d
