#!/usr/bin/env python
import os
import sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
d = pd.read_csv(os.path.join(P.RERUN_RESULTS, "metrics.csv"))
d = d[(d.seed == 0) & d.split.isin(["test", "train"]) & d.scene.isin(["gsqa/truck", "mip360/bonsai", "mip360/garden"])]
m = d.groupby(["scene", "loss", "variant", "split", "metric"]).value.mean()
rows = []
for scene in ("gsqa/truck", "mip360/bonsai", "mip360/garden"):
    for loss in ("dists", "flip", "lpips_vgg", "lpips_alex"):
        for var in ("pure", "mixed"):
            dte = m[(scene, loss, var, "test", loss)] - m[(scene, "default", "ctrl", "test", loss)]
            dtr = m[(scene, loss, var, "train", loss)] - m[(scene, "default", "ctrl", "train", loss)]
            rows.append(dict(scene=scene, metric=loss, variant=var, d_test=dte, d_train=dtr, G=dte / dtr))
r = pd.DataFrame(rows)
r.to_csv(os.path.join(P.S2_REC, "results", "g_reference.csv"), index=False)
p = r.pivot_table(index=["metric", "variant"], columns="scene", values="G")
L = ["**参考（未登记，仅作背景）：正常优化某个指标时的 G**（论文重跑实验里以该指标为损失训练的模型，相对对照模型，held-out 上的改善 ÷ 训练视角上的改善）", "",
     "| 被优化的指标 | 损失 | truck | bonsai | garden |", "|:--|:--|--:|--:|--:|"]
name = {"dists": "DISTS", "flip": "FLIP", "lpips_vgg": "LPIPS-VGG", "lpips_alex": "LPIPS-Alex"}
for (k, v), row in p.iterrows():
    L.append(f"| {name[k]} | {'单独' if v == 'pure' else '混合'} | {row['gsqa/truck']:.2f} | {row['mip360/bonsai']:.2f} | {row['mip360/garden']:.2f} |")
open(os.path.join(P.S2_REC, "results", "table_gref.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("range", round(r.G.min(), 2), round(r.G.max(), 2), "median", round(r.G.median(), 2))
