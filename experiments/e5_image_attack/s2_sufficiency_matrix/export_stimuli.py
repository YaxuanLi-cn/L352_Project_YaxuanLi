#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
sys.path.insert(0, HERE)
import judge as J              
RERUN = P.RERUN_STIMULI
OUT = P.data(P.E5, P.S2, "stimuli")
WIN = 512


def link(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.link(src, dst)


def best_window(a, b):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    h, w = d.shape
    wh, ww = min(WIN, h), min(WIN, w)
    ii = np.zeros((h + 1, w + 1), np.float64)
    ii[1:, 1:] = d.cumsum(0).cumsum(1)
    s = ii[wh:, ww:] - ii[:-wh, ww:] - ii[wh:, :-ww] + ii[:-wh, :-ww]
    y, x = np.unravel_index(int(np.argmax(s)), s.shape)                        
    return int(x), int(y), ww, wh, float(s[y, x] / (wh * ww)), float(d.mean())


def main():
    meta, means, cmean, views, ctrl_views = J.load()
    ver = pd.read_csv(os.path.join(P.S2_REC, "results", "verdicts.csv"))
    ver = ver[(ver.verdict != "no_zero_run") & ver.held.astype(bool)]
    dirs = meta.set_index("run").dir
    cv = ctrl_views[ctrl_views.res == "full"].pivot_table(index=["scene", "view"], columns="metric", values="value")
    rv = views[(views.res == "full") & (views.split == "test")].pivot_table(index=["run", "view"], columns="metric", values="value")
    rows = []
    for p in ver.itertuples():
        sdn = p.scene.replace("/", "_")
        point = f"{p.anchor}-{p.target}-{p.gamma:g}-s{p.seed}"
        for v in sorted(rv.loc[p.run].index):
            f = f"{int(v):03d}.png"
            src = dict(control=os.path.join(RERUN, sdn, "s0", "default__ctrl", "test", f), reference=os.path.join(RERUN, sdn, "reference", "test", f),
                       attacked=os.path.join(dirs[p.run], "test", f))
            full = dict(control=os.path.join(OUT, sdn, "control", "full", f), reference=os.path.join(OUT, sdn, "reference", "full", f),
                        attacked=os.path.join(OUT, sdn, point, "full", f))
            for k in src:
                link(src[k], full[k])
            img = {k: np.asarray(Image.open(src[k]).convert("RGB")) for k in src}
            x, y, ww, wh, mad_crop, mad_full = best_window(img["control"], img["attacked"])
            crop = {}
            for k in src:
                crop[k] = os.path.join(OUT, sdn, point, "crop", f"{int(v):03d}_{k}.png")
                if not os.path.exists(crop[k]):
                    os.makedirs(os.path.dirname(crop[k]), exist_ok=True)
                    Image.fromarray(img[k][y:y + wh, x:x + ww]).save(crop[k], compress_level=3)
            row = dict(scene=p.scene, view=int(v), held_set=p.anchor, held_set_label=J.ANCHOR_NAME[p.anchor], target=p.target, gamma=p.gamma, seed=p.seed,
                       source=p.source, verdict=p.verdict, verdict_label=J.VERDICT[p.verdict], verdict_strict=p.verdict_strict,
                       crop_x=x, crop_y=y, crop_w=ww, crop_h=wh, mean_abs_diff_crop_8bit=mad_crop, mean_abs_diff_full_8bit=mad_full,
                       **{f"{k}_full": os.path.relpath(full[k], OUT) for k in full}, **{f"{k}_crop": os.path.relpath(crop[k], OUT) for k in crop})
            for m in J.IMG:
                row[f"control_{m}"] = cv.loc[(p.scene, v), m]
                row[f"attacked_{m}"] = rv.loc[(p.run, v), m]
            rows.append(row)
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    s = man.groupby(["scene", "held_set", "target", "gamma", "seed", "verdict_label"]).agg(views=("view", "size"), mean_abs_diff_full=("mean_abs_diff_full_8bit", "mean"),
                                                                                        mean_abs_diff_crop=("mean_abs_diff_crop_8bit", "mean")).reset_index()
    s.to_csv(os.path.join(P.S2_REC, "results", "stimuli_summary.csv"), index=False)
    L = ["| 场景 | 守住的 | 目标 | γ | 种子 | 判定 | 成对图像数（视角） | 两张渲染图的平均绝对差：整图 / 裁剪窗口（8 位灰度级） |", "|:--|:--|:--|--:|--:|:--|--:|--:|"]
    order = {"psnr": 0, "default": 1, "triplet": 2}
    for r in s.assign(_s=s.scene.map(list(J.SCENE_NAME).index), _o=s.held_set.map(order)).sort_values(["_s", "_o", "target", "gamma", "seed"]).itertuples():
        L.append(f"| {J.SCENE_NAME[r.scene]} | {J.ANCHOR_NAME[r.held_set]} | {J.LABEL[r.target]} | {r.gamma:g} | {r.seed} | {r.verdict_label} | {r.views} | {r.mean_abs_diff_full:.2f} / {r.mean_abs_diff_crop:.2f} |")
    n = man.groupby("verdict_label").size().to_dict()
    L += ["", f"合计 {len(man)} 对（" + "，".join(f"{k} {v}" for k, v in n.items()) + "）。"]
    open(os.path.join(P.S2_REC, "results", "table_stimuli.md"), "w").write("\n".join(L) + "\n")
    print(s.round(2).to_string(index=False))
    print(len(man), "pairs;", man.groupby("verdict_label").size().to_dict())
if __name__ == "__main__":
    main()
