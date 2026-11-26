# E7 Is self-overrating concentrated in the uncertain regions of a scene

This attempt did not work out.

The hypothesis: where the scene is not pinned down by the data, each metric picks its own compromise, so the self-overrating
S from E4 should be concentrated in those regions.

1. Train 4 more default-loss 3DGS models per scene, which differ from the E4 base model only in the random seed. The
   per-pixel variance of their renders is the uncertainty map.
2. Cut the images into 192×192 tiles and split the tiles of each scene into 5 strata by uncertainty (Q1 most certain, Q5
   most uncertain).
3. For every fine-tuned model of the E4 rerun, compute 13 metrics on every tile, then recompute S inside each stratum.

The result does not support the hypothesis: S rises only a little from Q1 to Q5, and the most uncertain 20% of the tiles
hold only 22% of S. Detailed results are in `records/e7_uncertainty_strata/`.

## How to run

The stage A models of E4 and the 3DGS iteration ladder of E1 are needed first.

```bash
bash run_all.sh              # all steps (including the E4 rerun)
bash run_all.sh analysis     # skip training, rendering and the E4 rerun, only do the steps after tiling
```

What `run_all.sh` runs, in order:

```bash
python -m pytest -q tests ../../common/tests/test_uncertainty.py
python train_seeds.py configs/pilot.yaml      # GPU: train the seed models
python render_seeds.py configs/pilot.yaml ../e4_self_overrating/rerun/configs/rerun.yaml   # GPU
python tiles.py strata       # stratification
python tiles.py checks       # GPU: check the metric implementations
python tiles.py metrics      # GPU: per-tile metrics
python tiles.py ladder       # GPU: per-tile metrics on the E1 ladder
python tiles.py pixel        # GPU: pixel-level version
python tiles.py table        # build the long table
python analysis.py           # analysis and verdict
python figures.py            # figures
```

Every step skips results that already exist, so it can be restarted after an interruption. `launch_4gpu/` holds the scripts
we used to schedule the jobs on 4 GPUs. They are not needed to reproduce on one machine.
Seed models are in `data/experiments/e7_uncertainty_strata/ckpt_seeds/`, results in `records/e7_uncertainty_strata/pilot/`.
