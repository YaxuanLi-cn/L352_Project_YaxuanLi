# E4 Self-overrating: using a metric as the training loss of 3DGS

We take a quality metric, use it directly as the loss to fine-tune a trained 3DGS model, and check whether this metric rates
"its own" model higher than the other metrics do.

- Stage A: train one 3DGS per scene with the default loss (0.8 L1 + 0.2 D-SSIM) for 30000 steps. 8 scenes, 3 seeds for truck.
- Stage B: start from the stage A model, keep the set of Gaussians fixed, train another 10000 steps with a different loss,
  then compute 13 metrics on held-out views and on training views.

Each (scene, seed) has 18 runs: `base` without fine-tuning, the control `ctrl` that keeps the default loss, 9 `pure` runs that
use a single metric as the loss, and 7 `mixed` runs that add a metric loss to the default loss. That is 180 runs in total.
Self-overrating S = the gain of a model over `ctrl` on its own metric, minus its average gain on the other metrics. The exact
definition is in `original_run/analyse.py`.

The experiment was run twice. `original_run/` is the first time, where the fine-tuned models and the per-view renders were
not saved. `rerun/` repeats stage B with the same stage A models, seeds and weights, and saves the renders and the models.
The later experiments (E5–E8) all use the results of the rerun.

Training and the metrics are implemented in `code/common/nvsqa/gs.py` and `code/common/gh/`.
`gh/flip_torch.py` is copied from https://github.com/NVlabs/flip .

## How to run

Original run:

```bash
cd original_run
python run.py configs/full.yaml                          # GPU: stage A + 180 stage B runs
python baseline_check.py conventions configs/full.yaml   # GPU: recompute the base model metrics the way official 3DGS does
python extra_rows.py configs/full.yaml
python stimuli_manifest.py configs/full.yaml
python analyse.py configs/full.yaml                      # result tables
python figures.py configs/full.yaml                      # figures
python figure_examples.py configs/full.yaml --scene mip360/garden
```

Rerun (needs the stage A models and `runs/` of the original run):

```bash
cd rerun
python rerun_finetune.py configs/rerun.yaml   # GPU: 180 runs
python recompute_whole.py                     # recompute the result tables with ../original_run/analyse.py
python compare_whole.py                       # compare with the tables of the original run
python compare_renders.py                     # compare with the renders of the original run
```

Both `run.py` and `rerun_finetune.py` can be resumed and can run as several processes at once. `--list` only shows the
status of each run.
Models and renders are in `data/experiments/e4_self_overrating/`, result tables in `records/e4_self_overrating/`.
