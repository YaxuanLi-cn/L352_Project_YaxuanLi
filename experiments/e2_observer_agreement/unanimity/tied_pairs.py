#!/usr/bin/env python
import itertools
import os
import sys
import numpy as np
import pandas as pd
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
P90 = {"psnr": 0.45, "ssim": 0.008, "lpips_vgg": 0.011}
SETS = {"PSNR tied": ["psnr"], "PSNR + SSIM tied": ["psnr", "ssim"], "PSNR + SSIM + LPIPS tied": ["psnr", "ssim", "lpips_vgg"]}
rows = []
rng = np.random.default_rng(0)
for ds in ("gsqa", "nvsqa"):
    m = pd.read_csv(os.path.join(P.subjective_results(ds), "metrics.csv"))
    rec = []
    for scene, g in m.groupby("scene"):
        g = g.reset_index(drop=True)
        for i, j in itertools.combinations(range(len(g)), 2):
            r = dict(scene=scene, ddmos=abs(g.dmos[i] - g.dmos[j]))
            for k in P90:
                r[k] = abs(g[k][i] - g[k][j]) / P90[k]
            rec.append(r)
    p = pd.DataFrame(rec)
    scenes = p.scene.unique()
    for name, ks in SETS.items():
        sel = (p[ks] < 1).all(axis=1)
        q = p[sel]
        def stats(x, s=None):
            y = x[(x[ks] < 1).all(axis=1)]
            return (y.ddmos.mean(), (y.ddmos > 0.5).mean()) if len(y) else (np.nan, np.nan)
        bs = np.array([stats(pd.concat([p[p.scene == s] for s in rng.choice(scenes, len(scenes))])) for _ in range(1000)])
        rows.append(dict(dataset=ds, tied_on=name, n_pairs=len(q), share_of_pairs=len(q) / len(p), mean_abs_ddmos=q.ddmos.mean(),
                         lo=np.nanquantile(bs[:, 0], 0.025), hi=np.nanquantile(bs[:, 0], 0.975), share_ddmos_gt_0p5=(q.ddmos > 0.5).mean(),
                         lo2=np.nanquantile(bs[:, 1], 0.025), hi2=np.nanquantile(bs[:, 1], 0.975), max_ddmos=q.ddmos.max()))
    rows.append(dict(dataset=ds, tied_on="(all pairs)", n_pairs=len(p), share_of_pairs=1.0, mean_abs_ddmos=p.ddmos.mean(), share_ddmos_gt_0p5=(p.ddmos > 0.5).mean(), max_ddmos=p.ddmos.max()))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(P.UNANIMITY, "tied_pairs.csv"), index=False)
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
