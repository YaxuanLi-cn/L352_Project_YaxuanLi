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
SIGN = {"psnr": 1, "ssim": 1, "ms_ssim": 1, "lpips_vgg": -1, "lpips_alex": -1, "flip": -1, "cvvdp_image": 1}
TRIP = ["psnr", "ssim", "lpips_vgg"]
FIVE = ["psnr", "ssim", "lpips_alex", "flip", "cvvdp_image"]
B = 2000


def pair_class(a, b):
    if a.method != b.method:
        return "cross_method"
    full = {(a.ladder, a.level), (b.ladder, b.level)}
    same = a.ladder == b.ladder or (("iterations", 30000) in full and "views" in (a.ladder, b.ladder))
    return "same_ladder" if same else "cross_ladder"


def votes(g, pi, pj, names):
    s = np.stack([np.sign((g[m].to_numpy()[pi] - g[m].to_numpy()[pj]) * SIGN[m]) for m in names])
    pos = (s > 0).sum(0)
    return np.maximum(pos, len(names) - pos)
old = pd.read_csv(os.path.join(P.PART1_RESULTS, "conditions.csv"))
old = old[old.seed == 42].copy()
old["scene_key"] = old.dataset + "/" + old.scene
ft = pd.read_csv(os.path.join(P.RERUN_RESULTS, "metrics.csv"))
ft = ft[(ft.seed == 0) & (ft.split == "test") & ft.metric.isin(list(SIGN)) & (ft.variant != "base")]
ft["cond"] = ft.loss + "/" + ft.variant
ft = ft.groupby(["scene", "cond", "metric"]).value.mean().unstack("metric").reset_index()
scenes = sorted(set(old.scene_key) & set(ft.scene))
rec = []
for s in scenes:
    g, f = old[old.scene_key == s].reset_index(drop=True), ft[ft.scene == s].reset_index(drop=True)
    pi, pj = np.array(list(itertools.combinations(range(len(g)), 2))).T
    cls = np.array([pair_class(g.iloc[i], g.iloc[j]) for i, j in zip(pi, pj)])
    near = np.ones(len(pi), bool)
    for m in TRIP:
        near &= np.abs(g[m].to_numpy()[pi] - g[m].to_numpy()[pj]) <= np.ptp(f[m].to_numpy())
    fi, fj = np.array(list(itertools.combinations(range(len(f)), 2))).T
    isctrl = (f.cond.to_numpy()[fi] == "default/ctrl") | (f.cond.to_numpy()[fj] == "default/ctrl")
    sets = {("same_ladder", False): (g, pi[cls == "same_ladder"], pj[cls == "same_ladder"]),
            ("cross_ladder", False): (g, pi[cls == "cross_ladder"], pj[cls == "cross_ladder"]),
            ("cross_method", False): (g, pi[cls == "cross_method"], pj[cls == "cross_method"]),
            ("same_ladder", True): (g, pi[(cls == "same_ladder") & near], pj[(cls == "same_ladder") & near]),
            ("cross_ladder", True): (g, pi[(cls == "cross_ladder") & near], pj[(cls == "cross_ladder") & near]),
            ("cross_method", True): (g, pi[(cls == "cross_method") & near], pj[(cls == "cross_method") & near]),
            ("finetune (all pairs of the 17 models)", False): (f, fi, fj),
            ("finetune (loss model vs control)", False): (f, fi[isctrl], fj[isctrl])}
    for (c, matched), (tab, a, b) in sets.items():
        if len(a) < 5:
            continue
        v3, v5 = votes(tab, a, b, TRIP), votes(tab, a, b, FIVE)
        rec.append(dict(scene=s, pair_class=c, range_matched=matched, n_pairs=len(a), triplet_unanimous=float((v3 == 3).mean()),
                        five_5_0=float((v5 == 5).mean()), five_4_1=float((v5 == 4).mean()), five_3_2=float((v5 == 3).mean())))
per = pd.DataFrame(rec)
per.to_csv(os.path.join(P.UNANIMITY, "bridge_by_cause_by_scene.csv"), index=False)
rng = np.random.default_rng(0)
CAL = {"GS-QA": (0.816, 0.574), "NVS-QA": (0.850, 0.500)}
va = pd.read_csv(os.path.join(P.UNANIMITY, "vote_accuracy.csv"))
va = va[(va.subset == "all") & (va.pairs == "all pairs") & (va.vote == "triplet_vgg")].set_index("dataset")
CAL = {k: (float(va.acc_unanimous[k]), float(va.acc_split_majority[k])) for k in va.index}
rows = []
for (c, matched), q in per.groupby(["pair_class", "range_matched"], sort=False):
    x = q.set_index("scene").triplet_unanimous
    draws = rng.integers(0, len(x), (B, len(x)))
    bs = x.to_numpy()[draws].mean(1)
    row = dict(pair_class=c, range_matched=matched, n_scenes=len(x), n_pairs=int(q.n_pairs.sum()), triplet_unanimous=float(x.mean()),
               lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)), five_5_0=float(q.five_5_0.mean()),
               five_4_1=float(q.five_4_1.mean()), five_3_2=float(q.five_3_2.mean()))
    for k, (au, asp) in CAL.items():
        row[f"predicted_majority_accuracy_{k}"] = float(x.mean() * au + (1 - x.mean()) * asp)
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(P.UNANIMITY, "bridge_by_cause.csv"), index=False)
pd.set_option("display.width", 250)
print("calibration (unanimous, split):", CAL)
print(res.round(3).to_string(index=False))
