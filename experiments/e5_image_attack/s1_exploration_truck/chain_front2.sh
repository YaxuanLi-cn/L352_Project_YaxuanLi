#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
source /venv/rafel/bin/activate
python run_ft.py --scene gsqa/truck --loss front:triplet:dists:0 --init ctrl --steps 3000 --tag r1
python run_ft.py --scene gsqa/truck --loss front:psnr:dists:0 --init ctrl --steps 3000 --tag r1
for g in 0.05 0.2; do
  python run_ft.py --scene gsqa/truck --loss front:triplet:multi:$g --init ctrl --steps 3000 --tag r1
  python run_ft.py --scene gsqa/truck --loss front:psnr:multi:$g --init ctrl --steps 3000 --tag r1
done
