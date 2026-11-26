# E1 Degradation ladders

We train 3DGS and NeRF models on 8 scenes (kitchen, bonsai, garden, bicycle from Mip-NeRF 360; truck, train, playground2,
flowers from GS-QA) and make them worse along three ladders: number of iterations, number of training views, and the cap on
the number of Gaussians (3DGS only). Each scene has 30 conditions, 240 models in total. On the 4 GS-QA scenes part of the
conditions are trained again with a second random seed, which gives an estimate of the seed noise.
We then compute the metrics on all models and check whether the metrics rank them in the same order, and how much flicker
the per-frame metrics miss.

Training, rendering and the metrics are implemented in `code/common/nvsqa/`.

## How to run

First download Mip-NeRF 360 and GS-QA into `data/datasets/` with the scripts in `code/common/download/`.

```bash
SCENES="gsqa/truck gsqa/train mip360/kitchen mip360/bonsai mip360/garden mip360/bicycle gsqa/playground2 gsqa/flowers"

# 1. train and render (GPU); the two queues can run at the same time
python run_queue.py --scenes $SCENES --methods 3dgs &
python run_queue.py --scenes $SCENES --methods nerf &

# 2. second random seed (GPU), only on the 4 GS-QA scenes
SEEDS="3dgs-full@1 3dgs-views-4@1 3dgs-views-16@1 3dgs-gauss-30000@1 3dgs-gauss-300000@1 nerf-full@1 nerf-views-4@1 nerf-views-16@1"
python run_queue.py --scenes gsqa/truck gsqa/train gsqa/playground2 gsqa/flowers --jobs $SEEDS

# 3. compute the metrics (GPU); with --loop it can run alongside training
python eval_part1.py --loop

# 4. collect and analyse, after everything above has finished
python collect_part1.py      # conditions.csv, frames.csv
python describe_part1.py     # GPU, descriptors.csv
python analyse_part1.py      # all other tables
```

Finished jobs are skipped, so after an interruption just run the same command again.
Renders go to `data/experiments/e1_degradation_ladders/`, result tables to `records/e1_degradation_ladders/results/`.
