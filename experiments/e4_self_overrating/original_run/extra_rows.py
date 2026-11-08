#!/usr/bin/env python
import glob
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
ROOT = P.ROOT
import pandas as pd              
import torch              
import yaml              
cfg = yaml.safe_load(open(sys.argv[1]))
runs_root = os.path.join(ROOT, cfg["runs_root"])
stage_a = {}
for f in sorted(glob.glob(os.path.join(ROOT, cfg["ckpt_root"], "*", "base_seed*.pt"))):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    scene_dir = os.path.basename(os.path.dirname(f))
    stage_a[(scene_dir, ck["seed"] - 42)] = (float(ck["train_time"]), int(ck["steps"]))
    del ck
n = 0
for d in sorted(glob.glob(os.path.join(runs_root, "*", "s*", "*__*"))):
    info = json.load(open(os.path.join(d, "done.json")))
    key = dict(scene=info["scene"], loss=info["loss"], variant=info["variant"], seed=info["seed"])
    rows = [dict(key, split="model", view_id=-1, metric="eval_time_s", value=float(info["eval_time"]))]
    if info["variant"] == "base":
        t, steps = stage_a[(info["scene"].replace("/", "_"), info["seed"])]
        rows.append(dict(key, split="model", view_id=-1, metric="stage_a_train_time_s", value=t))
        rows.append(dict(key, split="model", view_id=-1, metric="stage_a_steps", value=float(steps)))
    pd.DataFrame(rows).to_csv(os.path.join(d, "extra_times.csv"), index=False)
    n += 1
print("extra rows written for", n, "runs;", len(stage_a), "stage-A checkpoints")
