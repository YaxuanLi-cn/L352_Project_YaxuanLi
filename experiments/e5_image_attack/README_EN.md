# E5 Image attack

Starting from the E4 rerun models, we fine-tune them as an attack: keep only the metrics that papers usually use, and see
whether the image quality measured by the other metrics can be pushed down. Everything is measured on held-out views.

The experiment has 6 stages, done in this order:

1. `s1_exploration_truck`: try different attack losses on truck.
2. `s2_sufficiency_matrix`: compute the seed-noise thresholds of the 13 metrics, run the attack matrix and judge the results.
3. `s3_tint_truck`: increase the DreamSim push step by step on truck, measure the colour shift against the control image,
   and build a comparison page.
4. `s4_tint_scenes`: repeat the colour shift experiment on the other scenes.
5. `s5_detectors`: compute the 13 metrics on the attacked models and the control models.
6. `s6_round2`: add DreamSim to the set of kept metrics and attack the metrics outside the set again.

## How to run

The E4 rerun models are needed first. `env.sh` in each stage directory sets the run directory and the record directory. Go
into the stage directory, source it, then run the scripts, for example:

```bash
cd code/experiments/e5_image_attack/s4_tint_scenes
source env.sh
python run_colour.py --scene gsqa/flowers --mult 1   # GPU: attack one scene
python colour.py                                     # compute the colour shift, writes levels.csv
```

The scripts that run the attacks are `run_ft.py`, `run_suff.py`, `run_tint.py`, `run_colour.py` and `run_ext.py` (GPU).
The other Python scripts do the summaries, the judging and the plots, and the `.sh` files are what we used for queueing.
Run outputs are in `data/experiments/e5_image_attack/<stage>/`, summaries in `records/e5_image_attack/<stage>/`.
