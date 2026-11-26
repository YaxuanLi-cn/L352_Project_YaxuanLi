# E8 Is self-overrating just random drift between two trainings

This attempt did not work out either.

Every configuration of E4 happens to be trained twice (original run and rerun, with the same stage A model, seed and loss),
but the two sets of renders are not the same. The idea to test: if the damage caused by changing the loss is mostly this
kind of drift that differs from training to training, then self-overrating is only noise.

The result is that random drift is not the main cause: most of the damage shows up again in both trainings, and S hardly
changes after averaging the renders of the two trainings.

The code has two parts. Both only use the renders that E4 already saved, no new model is trained:

- `drift.py`, `drift_analysis.py`, `drift_figures.py`: compute 13 metrics directly between the renders of the two trainings
  and look at how the size of the drift relates to S.
- `decomposition/`: split the squared error of one training into the part that both trainings reproduce and the part that is
  specific to each training.

## How to run

The original run and the rerun of E4 have to be finished. `drift_analysis.py` also needs `tiles_long.parquet` from E7.

```bash
# comparison of the two trainings
python drift.py pairs        # GPU
python drift_analysis.py
python drift_figures.py

# decomposition
cd decomposition
python step1_pixel.py gsqa/truck           # once per scene
python step2_metrics.py gsqa/truck 10 abm  # GPU: metrics on the mean image of the two trainings
python aggregate.py
python make_finding.py                     # writes FINDING.md and the figure
```

The three arguments of `step2_metrics.py` are the scene, the maximum number of views, and the mode (`abm` for GS-QA scenes,
`m` for Mip-NeRF 360 scenes).
Results are in `two_trainings/` and `decomposition/` under `records/e8_drift/`.
