#!/usr/bin/env python
import glob
import json
import math
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
sys.path.insert(0, P.code(P.E5, P.S2))
import judge as J              
RERUN = P.RERUN_STIMULI
OUT = os.path.join(P.S3_REC, "results")
SIGMA = 8.0
SCENES = list(J.SCENE_NAME)
VISIBLE_TRUCK = ("gsqa/truck", 0.0125)


def load_all():
    meta, means, cmean, views, ctrl_views = J.load()
    rows, frames = [], []
    for d in sorted(glob.glob(os.path.join(P.S3_RUNS, "*", "*"))):
        f = os.path.join(d, "done.json")
        if not os.path.exists(f):
            continue
        info = json.load(open(f))
        _, anchor, target, gamma = info["loss"].split(":")
        run = f"{info['scene'].replace('/', '_')}/{os.path.basename(d)}"
        rows.append(dict(run=run, scene=info["scene"], anchor=anchor, target=target, gamma=float(gamma), seed=int(info["seed"]), source="item11",
                         status=info["status"], train_time=info["train_time"], eval_time=info["eval_time"], wall_time=info["wall_time"], dir=d))
        if info["status"] != "ok":
            continue
        m = pd.read_parquet(os.path.join(d, "metrics.parquet"))
        frames.append(m[m.metric.isin(J.IMG)][["split", "view", "res", "metric", "value"]].assign(run=run))
    meta = pd.concat([meta, pd.DataFrame(rows)], ignore_index=True)
    views = pd.concat([views] + frames)
    means = views.groupby(["run", "split", "res", "metric"]).value.mean()
    return meta, means, cmean, views, ctrl_views


def stats(a_dir, b_dir, views):
    acc = []
    for v in views:
        a = cv2.imread(os.path.join(a_dir, f"{int(v):03d}.png")).astype(np.float32)
        b = cv2.imread(os.path.join(b_dir, f"{int(v):03d}.png")).astype(np.float32)
        d = a - b
        low = cv2.GaussianBlur(d, (0, 0), SIGMA)
        high = d - low
        luma = 0.114 * low[..., 0] + 0.587 * low[..., 1] + 0.299 * low[..., 2]
        chroma = low - luma[..., None]
        mx, cmx = np.abs(low).max(axis=2), np.abs(chroma).max(axis=2)
        acc.append(dict(rms_low=float(np.sqrt((low ** 2).mean())), rms_high=float(np.sqrt((high ** 2).mean())), p99_low=float(np.quantile(mx, 0.99)),
                        area_low_gt6=float((mx > 6).mean()), rms_low_chroma=float(np.sqrt((chroma ** 2).mean())),
                        area_chroma_gt6=float((cmx > 6).mean()), p99_chroma=float(np.quantile(cmx, 0.99)), rms_low_luma=float(np.sqrt((luma ** 2).mean()))))
    return pd.DataFrame(acc).mean().to_dict()


def levels(with_stats=True):
    T = pd.read_csv(P.THRESHOLDS).set_index("metric")
    thr, sm = T.p90.to_dict(), T.s_m.to_dict()
    meta, means, cmean, views, ctrl_views = load_all()
    ok = meta[meta.status == "ok"]
    held_views = views[views.split == "test"].groupby("run").view.unique()
    out, floor = [], []
    for scene in SCENES:
        zero = ok[(ok.scene == scene) & (ok.anchor == "triplet") & (ok.gamma == 0) & (ok.seed == 0)].iloc[0]
        lad = ok[(ok.scene == scene) & (ok.anchor == "triplet") & (ok.target == "dreamsim") & (ok.seed == 0)].sort_values("gamma")
        for row in lad.itertuples():
            c = J.classify(row, means, cmean, thr, zero.run)
            cz = J.classify(row, means, cmean, thr, zero.run, held_ref="zero")
            cur, z0 = means[row.run]["test"]["full"], means[zero.run]["test"]["full"]
            r = dict(scene=scene, gamma=row.gamma, source=row.source, run=row.run, dir=row.dir, zero_dir=zero.dir, verdict=c["verdict"], held=c["held"],
                     verdict_vs_gamma0=cz["verdict"], held_vs_gamma0=cz["held"], n_witness_families=c["n_witness_families"],
                     witness_families=c["witness_families"], **{f"held_{k}_over_p90": c["held_ratio"][k] for k in J.HELD["triplet"]},
                     **{f"held_vs_gamma0_{k}_over_p90": cz["held_ratio"][k] for k in J.HELD["triplet"]},
                     z_dreamsim=J.SIGN["dreamsim"] * (cur["dreamsim"] - z0["dreamsim"]) / sm["dreamsim"], dreamsim_over_p90=c["worse"]["dreamsim"] / thr["dreamsim"],
                     dreamsim_value=cur["dreamsim"], dreamsim_gamma0=z0["dreamsim"], flip_over_p90=c["worse"]["flip"] / thr["flip"],
                     d_cvvdp_jod=cur["cvvdp_image"] - z0["cvvdp_image"], cvvdp_over_p90=c["worse"]["cvvdp_image"] / thr["cvvdp_image"],
                     d_psnr_vs_control=cur["psnr"] - cmean[scene]["full"]["psnr"])
            if with_stats:
                r.update(stats(os.path.join(row.dir, "test"), os.path.join(zero.dir, "test"), sorted(held_views[row.run])))
            out.append(r)
        if with_stats:
            fl = ok[(ok.scene == scene) & (ok.anchor == "triplet") & (ok.target == "flip") & (ok.gamma == 0.05) & (ok.seed == 0)].iloc[0]
            floor.append(dict(scene=scene, run=fl.run, **stats(os.path.join(fl.dir, "test"), os.path.join(zero.dir, "test"), sorted(held_views[fl.run]))))
    return pd.DataFrame(out), pd.DataFrame(floor)


def bracket(g):
    nh = g[~g.held].gamma
    n = float(nh.min()) if len(nh) else None
    h = g[g.held & ((g.gamma < n) if n is not None else True)].gamma
    return (float(h.max()) if len(h) else None), n


def next_level(scene):
    lv, _ = levels(with_stats=False)
    h, n = bracket(lv[lv.scene == scene])
    if h is None or n is None:
        return None
    x = math.sqrt(h * n)
    return float(f"{x:.2g}")


def fmt_g(g):
    return f"{g:g}"


def main():
    os.makedirs(OUT, exist_ok=True)
    lv, floor = levels()
    fl = floor.set_index("scene")
    lv["chroma_over_floor"] = [r.rms_low_chroma / fl.rms_low_chroma[r.scene] for r in lv.itertuples()]
    vis = float(lv[(lv.scene == VISIBLE_TRUCK[0]) & np.isclose(lv.gamma, VISIBLE_TRUCK[1])].rms_low_chroma.iloc[0])
    lv["tint"] = lv.chroma_over_floor >= 3
    lv["tint_ge_truck_visible"] = lv.rms_low_chroma >= vis - 1e-9
    lv.drop(columns=["dir", "zero_dir"]).to_csv(os.path.join(OUT, "ladder.csv"), index=False)
    floor.to_csv(os.path.join(OUT, "floor.csv"), index=False)
    V = J.VERDICT
    L = ["**推力阶梯：三个都守、只攻击 DreamSim（每档一次训练，种子 0；分解相对同场景三元组 γ = 0 的运行，单位 8 位灰度级，20 个 held-out 视角的均值）**", "",
         "| 场景 | γ | 来源 | 三个数相对对照模型变差 ÷ 阈值（PSNR / SSIM / LPIPS-VGG；负 = 变好） | 判定（登记规则） | 以 γ = 0 为基准的判定（旁证） | DreamSim：z（÷ 阈值） | 超阈值的见证族数 | FLIP ÷ 阈值 | ColorVideoVDP (JOD) | "
         "低频 RMS | 高频 RMS | 低频差的 99% 分位 | 低频差 > 6 级的面积 | **纯色偏 RMS**（÷ 噪声底） | 纯色偏 > 6 级的面积 | 成片色偏 |",
         "|:--|--:|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for scene in SCENES:
        g = lv[lv.scene == scene].sort_values("gamma", ascending=False)
        for r in g.itertuples():
            mark = "有，且不小于 truck 可见档" if (r.tint and r.tint_ge_truck_visible) else ("有" if r.tint else "无")
            L.append(f"| {J.SCENE_NAME[scene]} | {fmt_g(r.gamma)} | {'第 10 条' if r.source != 'item11' else '**新**'} | "
                     f"{r.held_psnr_over_p90:+.1f} / {r.held_ssim_over_p90:+.1f} / {r.held_lpips_vgg_over_p90:+.1f} | {V[r.verdict]} | {V[r.verdict_vs_gamma0]} | "
                     f"{r.z_dreamsim:+.1f}（{r.dreamsim_over_p90:.1f}） | {r.n_witness_families} | {r.flip_over_p90:+.2f} | {r.d_cvvdp_jod:+.2f} | {r.rms_low:.2f} | {r.rms_high:.2f} | "
                     f"{r.p99_low:.1f} | {100 * r.area_low_gt6:.1f}% | **{r.rms_low_chroma:.2f}**（{r.chroma_over_floor:.1f}） | {100 * r.area_chroma_gt6:.1f}% | {mark} |")
        f = fl.loc[scene]
        L.append(f"| {J.SCENE_NAME[scene]} | 噪声底 | 攻击 FLIP（没有推动） | — | — | — | — | — | — | — | {f.rms_low:.2f} | {f.rms_high:.2f} | {f.p99_low:.1f} | {100 * f.area_low_gt6:.1f}% | "
                 f"**{f.rms_low_chroma:.2f}**（1.0） | {100 * f.area_chroma_gt6:.1f}% | — |")
    open(os.path.join(OUT, "table_ladder.md"), "w").write("\n".join(L) + "\n")
    ans, A = [], ["**问题 A：三个数守住的前提下，每个场景最多藏下多大的色偏**", "",
                  "| 场景 | 口径 | 守住的各档里纯色偏最大的一档：γ | 纯色偏 RMS（÷ 噪声底；÷ truck 可见档） | 低频差 > 6 级的面积 | 纯色偏 > 6 级的面积 | DreamSim z | 紧邻的守不住的一档：γ | 那一档的纯色偏 RMS / 面积 |",
                  "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for scene in SCENES:
        g = lv[lv.scene == scene]
        for key, name in (("held", "登记的主判定"), ("held_vs_gamma0", "以 γ = 0 为基准（旁证）")):
            h = g[g[key]]
            nh = g[~g[key]]
            n = nh.sort_values("gamma").iloc[0] if len(nh) else None
            if not len(h):
                A.append(f"| {J.SCENE_NAME[scene]} | {name} | 没有守住的档 | — | — | — | — | — | — |")
                continue
            b = h.sort_values("rms_low_chroma").iloc[-1]
            ans.append(dict(scene=scene, rule=key, gamma=b.gamma, rms_low_chroma=b.rms_low_chroma, chroma_over_floor=b.chroma_over_floor, chroma_over_truck_visible=b.rms_low_chroma / vis,
                            area_low_gt6=b.area_low_gt6, area_chroma_gt6=b.area_chroma_gt6, z_dreamsim=b.z_dreamsim, tint=bool(b.tint),
                            next_not_held_gamma=None if n is None else n.gamma, next_not_held_chroma=None if n is None else n.rms_low_chroma))
            A.append(f"| {J.SCENE_NAME[scene]} | {name} | {fmt_g(b.gamma)} | {b.rms_low_chroma:.2f}（{b.chroma_over_floor:.1f}；{b.rms_low_chroma / vis:.2f}） | {100 * b.area_low_gt6:.1f}% | "
                     f"{100 * b.area_chroma_gt6:.1f}% | {b.z_dreamsim:+.1f} | " + ("—" if n is None else f"{fmt_g(n.gamma)}") + " | " +
                     ("—" if n is None else f"{n.rms_low_chroma:.2f} / {100 * n.area_low_gt6:.1f}%") + " |")
    pd.DataFrame(ans).to_csv(os.path.join(OUT, "answer.csv"), index=False)
    A += ["", "**问题 B：bonsai、garden 上有没有\"三个数守住、同时出现成片色偏\"的一档**", ""]
    for scene in ("mip360/bonsai", "mip360/garden"):
        g = lv[lv.scene == scene]
        for key, name in (("held", "登记的主判定"), ("held_vs_gamma0", "以 γ = 0 为基准（旁证）")):
            hit = g[g[key] & g.tint].sort_values("gamma")
            h, n = (bracket(g) if key == "held" else bracket(g.assign(held=g.held_vs_gamma0)))
            gap = "" if (h is None or n is None) else f"守住的最强一档 γ = {fmt_g(h)}，守不住的最弱一档 γ = {fmt_g(n)}，相隔 {n / h:.2f} 倍"
            if len(hit):
                A.append(f"- {J.SCENE_NAME[scene]}（{name}）：**有**，γ = " + "、".join(fmt_g(x) for x in hit.gamma) + f"（纯色偏 RMS " +
                         "、".join(f"{x:.2f}" for x in hit.rms_low_chroma) + f"）。{gap}。")
            else:
                A.append(f"- {J.SCENE_NAME[scene]}（{name}）：**在采样到的档位里没有**。{gap}。")
    open(os.path.join(OUT, "table_answer.md"), "w").write("\n".join(A) + "\n")
    print("\n".join(L))
    print()
    print("\n".join(A))
if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "next":
        x = next_level(sys.argv[2])
        print("none" if x is None else f"{x:g}")
    else:
        main()
