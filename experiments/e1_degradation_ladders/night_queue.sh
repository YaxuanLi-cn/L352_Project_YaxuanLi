#!/bin/bash
cd "$(dirname "$0")"
source /venv/rafel/bin/activate
SCENES="gsqa/truck gsqa/train mip360/kitchen mip360/bonsai mip360/garden mip360/bicycle gsqa/playground2 gsqa/flowers"
SEEDJOBS="3dgs-views-4@1 3dgs-views-16@1 3dgs-gauss-30000@1 3dgs-gauss-300000@1 nerf-views-4@1 nerf-views-16@1 3dgs-full@1 nerf-full@1"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
python run_queue.py --scenes gsqa/train --jobs $SEEDJOBS
python run_queue.py --scenes $SCENES --methods 3dgs --reverse
while kill -0 "$2" 2>/dev/null; do sleep 30; done
python run_queue.py --scenes $SCENES --methods 3dgs nerf
python run_queue.py --scenes gsqa/playground2 gsqa/flowers --jobs $SEEDJOBS
echo "NIGHT QUEUE FINISHED $(date)"
