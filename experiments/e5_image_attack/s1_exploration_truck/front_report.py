#!/usr/bin/env python
import os
import re
import sys
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import numpy as np              
import pandas as pd              
REC = P.rec(P.E5, P.S1)
SEED_P90 = dict(psnr=0.45, ssim=0.008, lpips_vgg=0.011)
d = pd.read_csv(os.path.join(REC, "mad_summary_gsqa_truck.csv"))
d = d[d.run.str.startswith("front-")].copy()
m = d.run.str.extract(r"front-(?P<anchor>\w+?)-(?P<target>\w+)-(?P<gamma>[0-9.]+)__")
d = pd.concat([d, m], axis=1)
d["gamma"] = d.gamma.astype(float)
free = ["dists", "flip", "cvvdp_image", "dreamsim", "fsim", "vif"]
rows = []
for (anchor, target), g in d.groupby(["anchor", "target"]):
    zero = d[(d.anchor == anchor) & (d.gamma == 0)]
    z0 = zero.iloc[0]
    for _, r in g.sort_values("gamma").iterrows():
        row = dict(anchor=anchor, target=target, gamma=r.gamma)
        for k in ("psnr", "ssim", "lpips_vgg"):
            row[f"d_{k}"] = r[f"d_{k}"] - z0[f"d_{k}"]
            row[f"{k}_seedP90"] = (r[f"d_{k}"] - z0[f"d_{k}"]) / SEED_P90[k]
        for k in free + ["lpips_alex", "ms_ssim"]:
            row[f"z_{k}"] = r[f"z_{k}"] - z0[f"z_{k}"]
        row["d_cvvdp_jod"] = r.d_cvvdp_image - z0.d_cvvdp_image
        row["free_mean_z"] = float(np.mean([row[f"z_{k}"] for k in free]))
        worst = {"triplet": max(abs(row["psnr_seedP90"]), abs(row["ssim_seedP90"]), abs(row["lpips_vgg_seedP90"])), "psnr": abs(row["psnr_seedP90"]),
                 "default": max(abs(row["psnr_seedP90"]), abs(row["ssim_seedP90"]))}[anchor]
        row["reported_shift_seedP90"] = worst
        row["hidden_per_reported"] = -row["free_mean_z"] / worst if worst > 0 else np.nan
        rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(REC, "front_report_gsqa_truck.csv"), index=False)
pd.set_option("display.width", 250)
print(res[["anchor", "target", "gamma", "d_psnr", "d_ssim", "d_lpips_vgg", "reported_shift_seedP90", "z_dists", "z_flip", "z_cvvdp_image", "d_cvvdp_jod",
           "z_dreamsim", "z_fsim", "z_vif", "free_mean_z", "hidden_per_reported"]].round(3).to_string(index=False))
