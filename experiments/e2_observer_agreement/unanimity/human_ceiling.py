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
SIGN = {"psnr": 1, "ssim": 1, "lpips_vgg": -1, "lpips_alex": -1, "flip": -1, "cvvdp": 1}
m = pd.read_csv(os.path.join(P.NVSQA_RESULTS, "metrics.csv"))
s = pd.read_csv(os.path.join(P.NVSQA_RESULTS, "subjects.csv"))
obs = [c for c in s.columns if c.startswith("S") and s[c].notna().all()]
d = m.merge(s, on=["scene", "suffix"])
rows = []
for scene, g in d.groupby("scene"):
    g = g.reset_index(drop=True)
    for i, j in itertools.combinations(range(len(g)), 2):
        a, b = g.iloc[i], g.iloc[j]
        dd = a.dmos - b.dmos
        if dd == 0:
            continue
        v3 = [np.sign((a[k] - b[k]) * SIGN[k]) for k in ("psnr", "ssim", "lpips_vgg")]
        v5 = [np.sign((a[k] - b[k]) * SIGN[k]) for k in ("psnr", "ssim", "lpips_alex", "flip", "cvvdp")]
        sa, sb = a[obs].to_numpy(float), b[obs].to_numpy(float)
        n = len(obs)
        acc = []
        for o in range(n):
            others = (sa.sum() - sa[o] - sb.sum() + sb[o]) / (n - 1)
            own = sa[o] - sb[o]
            acc.append(0.5 if own == 0 or others == 0 else float(np.sign(own) == np.sign(others)))
        rows.append(dict(scene=scene, scene_class=a.scene_class, ddmos=abs(dd), unanimous3=abs(sum(v3)) == 3, maj3=np.sign(sum(v3)) == np.sign(dd),
                         majsize5=int(max(sum(x > 0 for x in v5), sum(x < 0 for x in v5))), maj5=np.sign(sum(v5)) == np.sign(dd),
                         one_observer=float(np.mean(acc)), frac_obs_with_majority=float(np.mean((np.sign(sa - sb) == np.sign(dd)) + 0.5 * (sa == sb)))))
p = pd.DataFrame(rows)
p.to_csv(os.path.join(P.UNANIMITY, "human_ceiling_pairs.csv"), index=False)
rng = np.random.default_rng(0)
scenes = p.scene.unique()


def boot(fn):
    v = []
    for _ in range(2000):
        q = pd.concat([p[p.scene == sc] for sc in rng.choice(scenes, len(scenes))])
        v.append(fn(q))
    return np.nanquantile(v, [0.025, 0.975])
out = []
for name, sel in (("all pairs", lambda q: q), ("triplet unanimous", lambda q: q[q.unanimous3]), ("triplet split", lambda q: q[~q.unanimous3]),
                  ("five families 5:0", lambda q: q[q.majsize5 == 5]), ("five families 4:1", lambda q: q[q.majsize5 == 4]),
                  ("five families 3:2", lambda q: q[q.majsize5 == 3])):
    q = sel(p)
    five = name.startswith("five")
    col = "maj5" if five else "maj3"
    lo_m, hi_m = boot(lambda x: sel(x)[col].mean())
    lo_h, hi_h = boot(lambda x: sel(x).one_observer.mean())
    lo_d, hi_d = boot(lambda x: sel(x)[col].mean() - sel(x).one_observer.mean())
    out.append(dict(pairs=name, n=len(q), metrics_majority_vs_dmos=q[col].mean(), m_lo=lo_m, m_hi=hi_m, one_observer_vs_others=q.one_observer.mean(),
                    h_lo=lo_h, h_hi=hi_h, metrics_minus_observer=q[col].mean() - q.one_observer.mean(), d_lo=lo_d, d_hi=hi_d,
                    mean_abs_ddmos=q.ddmos.mean()))
res = pd.DataFrame(out)
res.to_csv(os.path.join(P.UNANIMITY, "human_ceiling.csv"), index=False)
pd.set_option("display.width", 250)
print(len(obs), "observers with complete scores;", len(p), "pairs")
print(res.round(3).to_string(index=False))
