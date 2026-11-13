#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import colour as C
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
REC = P.S4_REC
GAP, BG = 16, (128, 128, 128)
rng = np.random.default_rng(20261125)
lv = pd.read_csv(os.path.join(REC, "results", "levels.csv"))
dirs = C.runs()
rows, skipped = [], []
for s in C.ALL:
    g = lv[(lv.scene == s) & lv.hidden]
    if s == "gsqa/truck":
        g = g[np.isclose(g.gamma, 0.0125)]
    if not len(g):
        skipped.append(C.NAME[s] + ("（没有跑）" if not (lv.scene == s).any() else ""))
        continue
    b = g.sort_values("gamma").iloc[-1]
    d = dirs[(dirs.scene == s) & np.isclose(dirs.gamma, b.gamma) & (dirs.source == b.source)].dir.iloc[0]
    hv = C.held_views(s)
    sdn = s.replace("/", "_")
    for n, i in enumerate((0, 4, 8, 12, 16)):
        v = hv[i]
        f = f"{v:03d}.png"
        ref = Image.open(os.path.join(C.RERUN, sdn, "reference", "test", f)).convert("RGB")
        ctrl = Image.open(os.path.join(C.ctrl_dir(s), f)).convert("RGB")
        att = Image.open(os.path.join(d, "test", f)).convert("RGB")
        w, h = ref.size
        left_attacked = bool(rng.random() < 0.5)
        canvas = Image.new("RGB", (2 * w + 3 * GAP, 2 * h + 3 * GAP), BG)
        canvas.paste(ref, ((canvas.width - w) // 2, GAP))
        canvas.paste(att if left_attacked else ctrl, (GAP, h + 2 * GAP))
        canvas.paste(ctrl if left_attacked else att, (w + 2 * GAP, h + 2 * GAP))
        name = f"{C.NAME[s]}_{n + 1}.png"
        canvas.save(os.path.join(REC, "show", name), compress_level=6)
        rows.append(dict(file=name, scene=C.NAME[s], level_times_gamma0=round(float(b.mult), 3), gamma=float(b.gamma), heldout_index=i, view=v,
                         attacked="left" if left_attacked else "right", source=b.source))
pd.DataFrame(rows).to_csv(os.path.join(REC, "show_answers", "answers.csv"), index=False)
open(os.path.join(REC, "results", "show.md"), "w").write(
    f"图：`show/`，共 {len(rows)} 张（" + "、".join(f"{k} {v} 张" for k, v in pd.DataFrame(rows).groupby('scene', sort=False).size().items()) + "）；"
    "每张上面是参考图，下面两张是候选（原始对照模型和被攻击模型），左右随机、图上不标。答案在 `show_answers/answers.csv`。" +
    (f"没有\"藏住了\"的档、因此没有出图的场景：{'、'.join(skipped)}。" if skipped else "") + "\n")
print(len(rows), "pictures;", "no hidden level:", skipped)
