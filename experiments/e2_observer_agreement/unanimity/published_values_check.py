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
SETS = {"published: PSNR-Y / SSIM / LPIPS": {"pub_PSNR-Y": 1, "pub_SSIM": 1, "pub_LPIPS": -1},
        "published: PSNR-YUV / MS-SSIM / LPIPS": {"pub_PSNR-YUV": 1, "pub_MS-SSIM": 1, "pub_LPIPS": -1},
        "published, six families: PSNR-Y / SSIM / LPIPS / DISTS / VIF / VMAF": {"pub_PSNR-Y": 1, "pub_SSIM": 1, "pub_LPIPS": -1, "pub_DISTS": -1, "pub_VIF": 1, "pub_VMAF": 1},
        "ours: PSNR / SSIM / LPIPS-VGG": {"psnr": 1, "ssim": 1, "lpips_vgg": -1}}
rows = []
rng = np.random.default_rng(0)
for ds in ("gsqa", "nvsqa"):
    m = pd.read_csv(os.path.join(P.subjective_results(ds), "metrics.csv"))
    for name, sign in SETS.items():
        rec = []
        for scene, g in m.groupby("scene"):
            g = g.reset_index(drop=True)
            for i, j in itertools.combinations(range(len(g)), 2):
                dd = g.dmos[i] - g.dmos[j]
                if dd == 0:
                    continue
                v = np.array([np.sign((g[c][i] - g[c][j]) * s) for c, s in sign.items()])
                pos = int((v > 0).sum())
                rec.append(dict(scene=scene, unanimous=pos in (0, len(v)), correct=np.sign(v.sum()) == np.sign(dd) if v.sum() != 0 else 0.5))
        p = pd.DataFrame(rec)
        scenes = p.scene.unique()
        def stat(q):
            return q[q.unanimous].correct.mean() - q[~q.unanimous].correct.mean()
        bs = [stat(pd.concat([p[p.scene == s] for s in rng.choice(scenes, len(scenes))])) for _ in range(1000)]
        rows.append(dict(dataset=ds, metrics=name, n_pairs=len(p), share_unanimous=p.unanimous.mean(), acc_unanimous=p[p.unanimous].correct.mean(),
                         acc_split_majority=p[~p.unanimous].correct.mean(), gap=stat(p), lo=np.nanquantile(bs, 0.025), hi=np.nanquantile(bs, 0.975)))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(P.UNANIMITY, "published_values_check.csv"), index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
