#!/bin/bash
. "$(dirname "$0")/env.sh" || exit 1
source /venv/rafel/bin/activate
python run_ft.py --scene gsqa/truck --loss madl:cvvdp_image:down --init ctrl --steps 2000 --tag r1
python run_ft.py --scene gsqa/truck --loss madl:cvvdp_image:up --init ctrl --steps 2000 --tag r1
