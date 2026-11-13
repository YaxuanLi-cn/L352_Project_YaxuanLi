#!/usr/bin/env python
import glob
import json
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
RERUN = P.RERUN_STIMULI
OUT = os.path.join(P.S4_REC, "results")
SCENES = ["gsqa/flowers", "gsqa/playground2", "gsqa/train", "mip360/kitchen", "mip360/bicycle", "mip360/garden", "mip360/bonsai"]                     
ALL = ["gsqa/truck"] + SCENES
NAME = {s: s.split("/")[1] for s in ALL}
HELD = ["psnr", "ssim", "lpips_vgg"]
SIGN = {"psnr": 1, "ssim": 1, "lpips_vgg": -1, "dreamsim": -1}
T_TRUCK, GAMMA_TRUCK = 0.0191150705019633, 0.0125
SIGMA = 8.0
FACTOR = 2.0                                        


def held_views(scene):
    v = json.load(open(os.path.join(P.RERUN_VIEWS, f"{scene.replace('/', '_')}.json")))
    h = np.asarray(v["heldout"])
    return [int(x) for x in h[np.unique(np.round(np.linspace(0, len(h) - 1, 20)).astype(int))]]


def ctrl_dir(scene):
    return os.path.join(RERUN, scene.replace("/", "_"), "s0", "default__ctrl", "test")


def stats(a_dir, b_dir, views):
    acc = []
    for v in views:
        a = cv2.imread(os.path.join(a_dir, f"{int(v):03d}.png")).astype(np.float32)
        b = cv2.imread(os.path.join(b_dir, f"{int(v):03d}.png")).astype(np.float32)
        d = a - b
        low = cv2.GaussianBlur(d, (0, 0), SIGMA)
        luma = 0.114 * low[..., 0] + 0.587 * low[..., 1] + 0.299 * low[..., 2]
        chroma = low - luma[..., None]
        mx, cmx = np.abs(low).max(axis=2), np.abs(chroma).max(axis=2)
        acc.append(dict(rms_low=float(np.sqrt((low ** 2).mean())), rms_high=float(np.sqrt(((d - low) ** 2).mean())), rms_chroma=float(np.sqrt((chroma ** 2).mean())),
                        area_low_gt6=float((mx > 6).mean()), area_chroma_gt6=float((cmx > 6).mean())))
    return pd.DataFrame(acc).mean().to_dict()


def floors():
    rows = []
    for s in ALL:
        sdn, hv = s.replace("/", "_"), held_views(s)
        rows.append(dict(scene=s, kind="existing model without attack (default + LPIPS-VGG)", **stats(os.path.join(RERUN, sdn, "s0", "lpips_vgg__mixed", "test"), ctrl_dir(s), hv)))
        z = os.path.join(P.S1_RUNS, "gsqa_truck", "front-triplet-dists-0__from-ctrl__3000__r1", "test") if s == "gsqa/truck" else \
            os.path.join(P.S2_RUNS, sdn, "front-triplet-dists-0__from-ctrl__3000__s0", "test")
        if os.path.isdir(z):
            rows.append(dict(scene=s, kind="item 10: triplet loss, gamma = 0, 3000 steps", **stats(z, ctrl_dir(s), hv)))
    f = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    f.to_csv(os.path.join(OUT, "floors.csv"), index=False)
    return f


def control_means():
    d = pd.read_csv(os.path.join(P.RERUN_RESULTS, "metrics.csv"))
    d = d[(d.loss == "default") & (d.variant == "ctrl") & (d.seed == 0) & (d.split == "test") & d.metric.isin(HELD + ["dreamsim"])]
    out = {}
    for s in ALL:
        x = d[(d.scene == s) & d.view_id.isin(held_views(s))]
        out[s] = x.groupby("metric").value.mean().to_dict()
    return out


def runs():
    rows = []
    for f in sorted(glob.glob(os.path.join(P.S4_RUNS, "*", "*", "done.json"))):
        j = json.load(open(f))
        if j["status"] != "ok":
            continue
        rows.append(dict(scene=j["scene"], gamma=j["gamma"], mult=j["mult"], gamma0=j["gamma0"], source="new", dir=os.path.dirname(f), start=j["dreamsim_start_train"]))
    for f in sorted(glob.glob(os.path.join(P.S2_RUNS, "*", "front-triplet-dreamsim-*__from-ctrl__3000__s0", "done.json"))):
        j = json.load(open(f))
        if j["status"] != "ok" or j["seed"] != 0:
            continue
        t0 = j["start_scales"]["_target_scale"]
        g0 = GAMMA_TRUCK * t0 / T_TRUCK
        rows.append(dict(scene=j["scene"], gamma=j["gamma"], mult=j["gamma"] / g0, gamma0=g0, source="item10", dir=os.path.dirname(f), start=t0))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    thr = pd.read_csv(P.THRESHOLDS).set_index("metric").p90.to_dict()
    fl = floors()
    floor = fl[fl.kind.str.startswith("existing")].set_index("scene")
    cm = control_means()
    out = []
    for r in runs().itertuples():
        m = pd.read_parquet(os.path.join(r.dir, "metrics.parquet"))
        m = m[(m.split == "test") & (m.res == "full") & m.metric.isin(HELD + ["dreamsim"])]
        hv = held_views(r.scene)
        mean = m.groupby("metric").value.mean()
        row = dict(scene=r.scene, gamma=r.gamma, mult=r.mult, gamma0=r.gamma0, source=r.source, dir=r.dir)
        for k in HELD:
            row[f"{k}_over_p90"] = -SIGN[k] * (mean[k] - cm[r.scene][k]) / thr[k]               
        row["hidden"] = all(row[f"{k}_over_p90"] < 1 for k in HELD)
        row["dreamsim_ratio"] = mean["dreamsim"] / cm[r.scene]["dreamsim"]
        row["dreamsim"] = mean["dreamsim"]
        row.update(stats(os.path.join(r.dir, "test"), ctrl_dir(r.scene), hv))
        row["floor_chroma"] = float(floor.rms_chroma[r.scene])
        row["chroma_over_floor"] = row["rms_chroma"] / row["floor_chroma"]
        row["clear"] = row["chroma_over_floor"] >= FACTOR
        out.append(row)
    res = pd.DataFrame(out).sort_values(["scene", "gamma"])
    res.drop(columns="dir").to_csv(os.path.join(OUT, "levels.csv"), index=False)
    L = ["| 场景 | 推力（× γ0） | γ | 来源 | PSNR / SSIM / LPIPS-VGG 变差 ÷ 阈值（负 = 变好） | 藏住了 | DreamSim ÷ 起点 | 纯色偏 RMS（÷ 噪声底） | 低频差 > 6 级的面积 | 纯色偏 > 6 级的面积 | 藏住了且色偏明显高于噪声底 |",
         "|:--|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|"]
    summ = []
    for s in ALL:
        g = res[res.scene == s]
        f = floor.loc[s]
        if s != "gsqa/truck":
            new3 = g[(g.source == "new")]
            summ.append(dict(scene=s, n_new_levels=len(new3), n_levels=len(g), any_hidden=bool(g.hidden.any()), hit=bool((g.hidden & g.clear).any()),
                             hit_3x=bool((g.hidden & (g.chroma_over_floor >= 3)).any()),
                             best_hidden_chroma=float(g[g.hidden].rms_chroma.max()) if g.hidden.any() else float("nan"),
                             best_hidden_ratio=float(g[g.hidden].chroma_over_floor.max()) if g.hidden.any() else float("nan")))
        if not len(g):
            L.append(f"| {NAME[s]} | — | — | 没跑 | — | — | — | — | — | — | — |")
        for r in g.itertuples():
            L.append(f"| {NAME[s]} | {r.mult:.2f} | {r.gamma:.4g} | {'本条新跑' if r.source == 'new' else '第 10 条已有'} | "
                     f"{r.psnr_over_p90:+.1f} / {r.ssim_over_p90:+.1f} / {r.lpips_vgg_over_p90:+.1f} | {'**是**' if r.hidden else '否'} | {r.dreamsim_ratio:.1f} 倍 | "
                     f"{r.rms_chroma:.2f}（{r.chroma_over_floor:.1f}） | {100 * r.area_low_gt6:.1f}% | {100 * r.area_chroma_gt6:.1f}% | {'**是**' if (r.hidden and r.clear) else '否'} |")
        L.append(f"| {NAME[s]} | 噪声底 | — | 已有的不加攻击的模型 | — | — | — | {f.rms_chroma:.2f}（1.0） | {100 * f.area_low_gt6:.1f}% | {100 * f.area_chroma_gt6:.1f}% | — |")
    open(os.path.join(OUT, "table_levels.md"), "w").write("\n".join(L) + "\n")
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    done = sm[sm.n_new_levels == 3]
    part = sm[(sm.n_new_levels > 0) & (sm.n_new_levels < 3)]
    none = sm[sm.n_new_levels == 0]
    hits = sm[sm.hit]
    S = [f"7 个场景里有 **{len(hits)} 个**存在\"藏住了且色偏明显高于噪声底\"的档" + (f"（{'、'.join(NAME[s] for s in hits.scene)}）" if len(hits) else "") +
         f"；\"明显高于\"按纯色偏 RMS 不小于该场景噪声底的 {FACTOR:g} 倍计，若按 3 倍计是 {int(sm.hit_3x.sum())} 个" +
         (f"（{'、'.join(NAME[s] for s in sm[sm.hit_3x].scene)}）" if sm.hit_3x.any() else "") + "。",
         f"三档都跑完的场景 {len(done)} 个（{'、'.join(NAME[s] for s in done.scene) or '无'}）；" +
         (f"只跑完部分档的 {len(part)} 个（{'、'.join(NAME[s] for s in part.scene)}）；" if len(part) else "") +
         f"本条没有新跑的 {len(none)} 个（{'、'.join(NAME[s] for s in none.scene) or '无'}）。"]
    tr = res[(res.scene == "gsqa/truck") & res.hidden]
    S.append("各场景\"藏住了\"的档里纯色偏最大是噪声底的几倍：" + "、".join(
        f"{NAME[r.scene]} {r.best_hidden_ratio:.1f}" + ("（只有第 10 条已有的档）" if r.n_new_levels == 0 else "") for r in sm.itertuples() if r.any_hidden) +
        (f"；没有任何结果的场景：{'、'.join(NAME[r.scene] for r in sm.itertuples() if r.n_levels == 0)}" if (sm.n_levels == 0).any() else "") +
        (f"。对照：truck 是 {tr.chroma_over_floor.max():.1f} 倍。" if len(tr) else "。"))
    open(os.path.join(OUT, "summary.md"), "w").write("\n\n".join(S) + "\n")
    F = ["| 场景 | 噪声底的来源 | 纯色偏 RMS | 低频差 > 6 级的面积 | 纯色偏 > 6 级的面积 |", "|:--|:--|--:|--:|--:|"]
    for r in fl.itertuples():
        F.append(f"| {NAME[r.scene]} | {'已有的不加攻击的模型（默认损失 + LPIPS-VGG，表中所用）' if r.kind.startswith('existing') else '第 10 条的三元组 γ = 0 运行（仅供对照）'} | {r.rms_chroma:.2f} | {100 * r.area_low_gt6:.1f}% | {100 * r.area_chroma_gt6:.1f}% |")
    open(os.path.join(OUT, "table_floors.md"), "w").write("\n".join(F) + "\n")
    print("\n".join(L)); print(); print("\n".join(S)); print(); print("\n".join(F))
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "floors":
        pd.set_option("display.width", 200)
        print(floors().round(3).to_string(index=False))
    else:
        main()
