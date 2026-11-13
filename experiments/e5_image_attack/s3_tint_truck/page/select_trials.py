#!/usr/bin/env python
import json
import os
import sys
import shutil
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
PAGE = P.rec(P.E5, P.S3, "page")                                                              
S = P.data(P.E5, P.S2, "stimuli")
man = pd.read_csv(os.path.join(S, "manifest.csv"))
SCENE = {"gsqa/truck": "truck", "mip360/bonsai": "bonsai", "mip360/garden": "garden"}
TARGET = {"dists": "DISTS", "dreamsim": "DreamSim", "flip": "FLIP", "multi": "多目标（主要是 DreamSim）"}
SHOW = ["psnr", "ssim", "lpips_vgg", "dists", "dreamsim", "flip", "cvvdp_image"]
trials = []
for key, g in man.groupby(["scene", "held_set", "target", "gamma", "seed"], sort=True):
    v = g.verdict.iloc[0]
    if v == "real" and not (key[1] == "psnr" and key[2] == "dists" and key[3] == 0.05):
        continue
    g = g.sort_values("mean_abs_diff_crop_8bit").reset_index(drop=True)
    ranks = [19, 14, 9] if v == "suspect" else [19, 9]
    for rk in ranks:
        r = g.iloc[rk]
        tid = f"t{len(trials) + 1:02d}"
        files = {}
        for k, short in (("control", "c"), ("attacked", "a"), ("reference", "r")):
            dst = os.path.join(PAGE, "img", f"{tid}_{short}.png")
            shutil.copyfile(os.path.join(S, r[f"{k}_crop"]), dst)
            files[short] = f"img/{tid}_{short}.png"
        trials.append(dict(id=tid, scene=SCENE[r.scene], held=r.held_set_label, target=TARGET[r.target], gamma=float(r.gamma), seed=int(r.seed), verdict=r.verdict_label,
                           view=int(r["view"]), rank=rk + 1, diff=round(float(r.mean_abs_diff_crop_8bit), 2), **files,
                           m={k: [round(float(r[f"control_{k}"]), 4), round(float(r[f"attacked_{k}"]), 4)] for k in SHOW}))
json.dump(trials, open(os.path.join(PAGE, "trials.json"), "w"), ensure_ascii=False)
t = pd.DataFrame(trials)
print(t.groupby("verdict").size().to_dict(), len(t), "trials")
print(t.groupby(["scene", "held", "target", "gamma", "seed", "verdict"]).size().to_string())
