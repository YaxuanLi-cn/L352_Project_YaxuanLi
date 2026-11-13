#!/usr/bin/env python
import itertools
import os
import sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
IMG = ["l1", "mse", "psnr", "ssim", "ms_ssim", "lpips_vgg", "lpips_alex", "dists", "flip", "cvvdp_image", "dreamsim", "fsim", "vif"]
old = pd.read_csv(os.path.join(P.PART1_RESULTS, "seed_noise.csv")).select_dtypes("number").quantile(0.9)
sm = pd.read_csv(os.path.join(P.ORIG_TABLES, "column_scale.csv")).set_index("metric").scale
d = pd.read_csv(os.path.join(P.RERUN_RESULTS, "metrics.csv"))
t = d[(d.scene == "gsqa/truck") & (d.split == "test") & d.metric.isin(IMG) & (d.variant != "base")]
m = t.groupby(["loss", "variant", "seed", "metric"]).value.mean().unstack("seed")
pairs = pd.concat([(m[a] - m[b]).abs() for a, b in itertools.combinations([0, 1, 2], 2)])
rerun = pairs.groupby("metric").quantile(0.9)
common = [k for k in IMG if k in old.index]
factor = float((old[common] / rerun[common]).median())
alt_factor = float((old[common] / sm[common]).median())
rows = []
for k in IMG:
    has = k in old.index
    rows.append(dict(metric=k, source="old seed table" if has else "truck re-run seeds x factor", p90=float(old[k]) if has else float(rerun[k] * factor),
                     old_p90=float(old[k]) if has else float("nan"), rerun_truck_p90=float(rerun[k]), s_m=float(sm[k]),
                     p90_alt_fill=float(old[k]) if has else float(alt_factor * sm[k])))
res = pd.DataFrame(rows)
res["p90_in_s_m"] = res.p90 / res.s_m
res["factor_old_over_rerun"] = factor
res["n_pairs_rerun"] = len(pairs) // len(IMG)
res.to_csv(P.THRESHOLDS, index=False)
lines = ["| 指标 | 阈值（种子噪声 P90，原生单位） | 折合 s_m | 来源 | 另一种补法（仅作敏感性检查） |", "|:--|--:|--:|:--|--:|"]
for r in res.itertuples():
    lines.append(f"| {r.metric} | {r.p90:.4g} | {r.p90_in_s_m:.2f} | {'旧项目种子表' if r.source.startswith('old') else 'truck 三个种子 × ' + format(factor, '.2f')} | {r.p90_alt_fill:.4g} |")
open(os.path.join(P.S2_REC, "thresholds.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"factor old/rerun (median over {len(common)} metrics): {factor:.3f}; alt factor old/s_m: {alt_factor:.3f}; pairs: {len(pairs) // len(IMG)}")
