#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
sys.path.insert(0, HERE)
import judge as J              
RERUN = P.RERUN_STIMULI
SIGMA = 8.0


def stats(a_dir, b_dir, views):
    acc = []
    for v in views:
        a = cv2.imread(os.path.join(a_dir, f"{int(v):03d}.png")).astype(np.float32)
        b = cv2.imread(os.path.join(b_dir, f"{int(v):03d}.png")).astype(np.float32)
        d = a - b                                                     
        low = cv2.GaussianBlur(d, (0, 0), SIGMA)
        high = d - low
        luma = 0.114 * low[..., 0] + 0.587 * low[..., 1] + 0.299 * low[..., 2]
        mx = np.abs(low).max(axis=2)
        acc.append(dict(rms_low=float(np.sqrt((low ** 2).mean())), rms_high=float(np.sqrt((high ** 2).mean())), p99_low=float(np.quantile(mx, 0.99)),
                        area_low_gt6=float((mx > 6).mean()), rms_low_chroma=float(np.sqrt(((low - luma[..., None]) ** 2).mean()))))
    return pd.DataFrame(acc).mean().to_dict()


def main():
    meta, means, cmean, views, ctrl_views = J.load()
    ok = meta[meta.status == "ok"]
    ver = pd.read_csv(os.path.join(P.S2_REC, "results", "verdicts.csv")).set_index("run")
    held = views[views.split == "test"].groupby("run").view.unique()
    rows = []
    for r in ok.itertuples():
        sdn = r.scene.replace("/", "_")
        if r.gamma == 0:
            if r.seed == 0:
                rows.append(dict(scene=r.scene, anchor=r.anchor, target="（γ = 0，相对原始对照模型）", gamma=0.0, seed=r.seed, verdict="",
                                 **stats(os.path.join(r.dir, "test"), os.path.join(RERUN, sdn, "s0", "default__ctrl", "test"), sorted(held[r.run]))))
            continue
        z = ok[(ok.scene == r.scene) & (ok.anchor == r.anchor) & (ok.gamma == 0) & (ok.seed == 0)]
        rows.append(dict(scene=r.scene, anchor=r.anchor, target=r.target, gamma=r.gamma, seed=r.seed, verdict=ver.verdict.get(r.run, ""),
                         **stats(os.path.join(r.dir, "test"), os.path.join(z.iloc[0].dir, "test"), sorted(held[r.run]))))
    cdir = os.path.join(P.S2_AUX, "controls_png")
    crows = []
    if os.path.isdir(cdir):                                           
        cm = pd.read_csv(os.path.join(P.S2_REC, "results", "controls.csv"))
        cm = cm[cm.scene == "gsqa/truck"].set_index("condition")
        hv = sorted(int(f[:3]) for f in os.listdir(os.path.join(cdir, "tint")))
        for k in ("blur15", "tint", "adv_dists", "adv_dreamsim", "adv_flip"):
            crows.append(dict(condition=k, z_dreamsim=cm.z_dreamsim[k], z_dists=cm.z_dists[k], z_flip=cm.z_flip[k],
                              **stats(os.path.join(cdir, k), os.path.join(RERUN, "gsqa_truck", "s0", "default__ctrl", "test"), hv)))
        pd.DataFrame(crows).to_csv(os.path.join(P.S2_REC, "results", "diff_lowpass_controls.csv"), index=False)
    res = pd.DataFrame(rows)
    res["_o"] = res.anchor.map({"psnr": 0, "default": 1, "triplet": 2})
    res = res.sort_values(["scene", "_o", "gamma", "target", "seed"]).drop(columns="_o")
    res.to_csv(os.path.join(P.S2_REC, "results", "diff_lowpass.csv"), index=False)
    L = ["**画面变化的低频 / 高频分解（未登记的补充读数）**：被攻击模型与同一守住集合、γ = 0 的模型的 held-out 渲染图之差，分成低通部分（高斯 σ = 8 px）和其余部分；单位是 8 位灰度级，20 个视角的均值",
         "", "| 场景 | 守住的 | 目标 | γ | 种子 | 判定 | 低频 RMS | 高频 RMS | 低频差的 99% 分位 | 低频差 > 6 级的面积 | 低频差里纯色偏的 RMS |", "|:--|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|"]
    for r in res.itertuples():
        L.append(f"| {J.SCENE_NAME[r.scene]} | {J.ANCHOR_NAME[r.anchor]} | {J.LABEL.get(r.target, r.target)} | {r.gamma:g} | {r.seed} | {J.VERDICT.get(r.verdict, '—')} | "
                 f"{r.rms_low:.2f} | {r.rms_high:.2f} | {r.p99_low:.1f} | {100 * r.area_low_gt6:.1f}% | {r.rms_low_chroma:.2f} |")
    if crows:
        name = {"blur15": "真实退化：高斯模糊 σ = 1.5 px", "tint": "真实退化：低频色偏 ±6/255", "adv_dists": "二维对抗样本：攻击 DISTS（±2/255）",
                "adv_dreamsim": "二维对抗样本：攻击 DreamSim（±2/255）", "adv_flip": "二维对抗样本：攻击 FLIP（±2/255）"}
        L += ["", "**同一分解用在校准图像上（truck，相对未处理的对照模型渲染图）**", "",
              "| 处理 | DreamSim z | DISTS z | FLIP z | 低频 RMS | 高频 RMS | 低频差的 99% 分位 | 低频差 > 6 级的面积 | 低频差里纯色偏的 RMS |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in crows:
            L.append(f"| {name[r['condition']]} | {r['z_dreamsim']:+.1f} | {r['z_dists']:+.1f} | {r['z_flip']:+.1f} | {r['rms_low']:.2f} | {r['rms_high']:.2f} | {r['p99_low']:.1f} | "
                     f"{100 * r['area_low_gt6']:.1f}% | {r['rms_low_chroma']:.2f} |")
    open(os.path.join(P.S2_REC, "results", "table_lowpass.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
if __name__ == "__main__":
    main()
