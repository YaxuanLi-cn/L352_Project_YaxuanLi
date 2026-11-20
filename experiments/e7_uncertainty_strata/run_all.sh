#!/bin/bash
set -e
. "$(dirname "$0")/launch_4gpu/env.sh"
P=$E7/configs/pilot.yaml
R=$E4/configs/rerun.yaml
python -m pytest -q code/common/tests/test_uncertainty.py $E7/tests
if [ "$1" != "analysis" ]; then
  python $E7/train_seeds.py $P
  python $E7/render_seeds.py $P $R
  python $E4/rerun_finetune.py $R
  python $E4/recompute_whole.py
  python $E4/compare_whole.py
  python $E4/compare_renders.py
fi
python $E7/tiles.py strata
python $E7/tiles.py checks
python $E7/tiles.py metrics
python $E7/tiles.py ladder
python $E7/tiles.py pixel
python $E7/tiles.py table
python $E7/analysis.py
python $E7/figures.py
