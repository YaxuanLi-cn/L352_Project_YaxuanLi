#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
source /venv/rafel/bin/activate
for g in 0.05 0.2 0.02 0.1; do
  python run_ft.py --scene gsqa/truck --loss front:triplet:dists:$g --init ctrl --steps 3000 --tag r1
  python run_ft.py --scene gsqa/truck --loss front:psnr:dists:$g --init ctrl --steps 3000 --tag r1
done
python run_ft.py --scene gsqa/truck --loss ctrl --init ctrl --steps 3000 --tag r1
