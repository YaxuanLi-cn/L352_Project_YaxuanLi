#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
S = P.data(P.E5, P.S2, "stimuli")
man = pd.read_csv(os.path.join(S, "manifest.csv"))
os.makedirs(os.path.join(P.S2_REC, "figs"), exist_ok=True)
rows = []
for key, g in man.groupby(["scene", "held_set", "target", "gamma", "seed"]):
    g = g.sort_values("mean_abs_diff_crop_8bit").reset_index(drop=True)
    for tag, r in (("typical", g.iloc[len(g) // 2]), ("worst", g.iloc[-1])):
        im = {k: np.asarray(Image.open(os.path.join(S, r[f"{k}_crop"])).convert("RGB")) for k in ("reference", "control", "attacked")}
        diff = np.clip(np.abs(im["control"].astype(np.int16) - im["attacked"].astype(np.int16)) * 8, 0, 255).astype(np.uint8)
        strip = np.concatenate([im["reference"], im["control"], im["attacked"], diff], axis=1)
        img = Image.fromarray(strip)
        d = ImageDraw.Draw(img)
        w = im["control"].shape[1]
        for i, t in enumerate(("reference", "control", f"attacked ({r.held_set}:{r.target}, gamma {r.gamma:g})", "|control - attacked| x 8")):
            d.rectangle([i * w, 0, i * w + 8 * len(t) + 8, 16], fill=(0, 0, 0))
            d.text((i * w + 4, 2), t, fill=(255, 255, 255))
        name = f"{r.scene.replace('/', '_')}__{r.held_set}-{r.target}-{r.gamma:g}-s{r.seed}__{tag}_v{int(r['view']):03d}.png"
        img.save(os.path.join(P.S2_REC, "figs", name), compress_level=6)
        rows.append(dict(scene=r.scene, held_set=r.held_set, target=r.target, gamma=r.gamma, seed=r.seed, verdict=r.verdict_label, which=tag, view=int(r["view"]),
                         mean_abs_diff_crop_8bit=r.mean_abs_diff_crop_8bit, file=os.path.join("figs", name)))
pd.DataFrame(rows).to_csv(os.path.join(P.S2_REC, "results", "figs_index.csv"), index=False)
print(len(rows), "strips")
