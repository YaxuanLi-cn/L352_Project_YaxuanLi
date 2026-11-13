#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
source /venv/rafel/bin/activate
python run_ft.py --scene gsqa/truck --loss madl:dists:down --init ctrl --steps 3000 --tag r1
python run_ft.py --scene gsqa/truck --loss madl:dists:up --init ctrl --steps 3000 --tag r1
python run_ft.py --scene gsqa/truck --loss ctrl --init ctrl --steps 3000 --tag r1
python run_ft.py --scene gsqa/truck --loss madl:dists:down --hold psnr --init ctrl --steps 3000 --tag r1
