#!/usr/bin/env python
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
OUT = os.path.join(P.S2_REC, "results")
IMG = ["l1", "mse", "psnr", "ssim", "ms_ssim", "lpips_vgg", "lpips_alex", "dists", "flip", "cvvdp_image", "dreamsim", "fsim", "vif"]
SIGN = {"l1": -1, "mse": -1, "psnr": 1, "ssim": 1, "ms_ssim": 1, "lpips_vgg": -1, "lpips_alex": -1, "dists": -1, "flip": -1,
        "cvvdp_image": 1, "dreamsim": -1, "fsim": 1, "vif": 1}
LABEL = {"l1": "L1", "mse": "MSE", "psnr": "PSNR", "ssim": "SSIM", "ms_ssim": "MS-SSIM", "lpips_vgg": "LPIPS-VGG", "lpips_alex": "LPIPS-Alex",
         "dists": "DISTS", "flip": "FLIP", "cvvdp_image": "ColorVideoVDP", "dreamsim": "DreamSim", "fsim": "FSIM", "vif": "VIF", "multi": "多目标"}
PIXEL = {"l1", "mse", "psnr"}
TERMS = {"psnr": {"mse"}, "default": {"l1", "ssim"}, "triplet": {"l1", "ssim", "lpips_vgg"}}
HELD = {"psnr": ["psnr"], "default": ["psnr", "ssim"], "triplet": ["psnr", "ssim", "lpips_vgg"]}
WFAM = {"ssim": "ssim", "ms_ssim": "ssim", "lpips_vgg": "vgg16", "dists": "vgg16", "lpips_alex": "alexnet", "dreamsim": "dino_clip",
        "flip": "flip", "vif": "vif", "fsim": "fsim", "cvvdp_image": "cvvdp"}
BACKBONE = {"lpips_vgg": "vgg16", "dists": "vgg16", "lpips_alex": "alexnet", "dreamsim": "dino_clip"}
MULTI = ["dists", "flip", "dreamsim", "vif"]
ANCHOR_NAME = {"psnr": "只守 PSNR", "default": "守 PSNR + SSIM", "triplet": "守 PSNR + SSIM + LPIPS"}
SCENE_NAME = {"gsqa/truck": "truck", "mip360/bonsai": "bonsai", "mip360/garden": "garden"}
VERDICT = {"not_held": "守不住", "cannot_push": "推不动", "real": "真实退化", "suspect": "疑似对抗样本"}
WCOLS = ["cvvdp_image", "fsim", "vif", "flip", "dreamsim", "lpips_alex", "dists", "lpips_vgg", "ssim", "ms_ssim"]
REPRO = []


def load():
    meta, frames = [], []
    for d in sorted(glob.glob(os.path.join(P.S1_RUNS, "gsqa_truck", "front-*"))) + sorted(glob.glob(os.path.join(P.S2_RUNS, "*", "*"))):
        f = os.path.join(d, "done.json")
        if not os.path.exists(f):
            continue
        info = json.load(open(f))
        _, anchor, target, gamma = info["loss"].split(":")
        new = d.startswith(P.S2_RUNS)
        sdn = info["scene"].replace("/", "_")
        run = f"{sdn}/{os.path.basename(d)}"
        meta.append(dict(run=run, scene=info["scene"], anchor=anchor, target=target, gamma=float(gamma), seed=int(info.get("seed", 0)),
                         source="new" if new else "item9", status=info["status"], train_time=info["train_time"], eval_time=info.get("eval_time", np.nan),
                         wall_time=info.get("wall_time", np.nan), dir=d))
        if info["status"] != "ok":
            continue
        m = pd.read_parquet(os.path.join(d, "metrics.parquet"))
        m = m[m.metric.isin(IMG)].copy()
        if not new:
            m["res"] = "stored"
            a = os.path.join(P.S2_AUX, f"{sdn}__{os.path.basename(d)}.parquet")
            a = pd.read_parquet(a)
            chk = a[a.res == "full"].set_index(["view", "metric"]).value - m[m.split == "test"].set_index(["view", "metric"]).value
            REPRO.append(pd.DataFrame({"run": run, "max_abs": chk.abs().groupby("metric").max(), "mean_diff": chk.groupby("metric").mean()}).reset_index())
            m = pd.concat([m, a])
        frames.append(m[["split", "view", "res", "metric", "value"]].assign(run=run))
    meta = pd.DataFrame(meta)
    views = pd.concat(frames)
    means = views.groupby(["run", "split", "res", "metric"]).value.mean()
    ctrl = []
    for f in sorted(glob.glob(os.path.join(P.S2_AUX, "*__control.parquet"))):
        ctrl.append(pd.read_parquet(f))
    ctrl = pd.concat(ctrl)
    return meta, means, ctrl.groupby(["scene", "res", "metric"]).value.mean(), views, ctrl


def classify(row, means, cmean, thr, zero_run, strict=False, held_ref="control"):
    a, tgt = row.anchor, row.target
    attacked = MULTI if tgt == "multi" else [tgt]
    cur = means[row.run]["test"]["full"]
    zero = means[zero_run]["test"]["full"]
    con = cmean[row.scene]["full"]
    out = dict(attacked=",".join(attacked))
    base = con if held_ref == "control" else zero
    held_ratio = {k: -SIGN[k] * (cur[k] - base[k]) / thr[k] for k in HELD[a]}
    worse = {k: -SIGN[k] * (cur[k] - zero[k]) for k in IMG}
    held = all(v < 1 for v in held_ratio.values())
    moved = any(worse[k] >= thr[k] for k in attacked)
    bb = {BACKBONE[k] for k in attacked if k in BACKBONE}
    wit, excluded = [], {}
    for k in IMG:
        if k in PIXEL:
            excluded[k] = "pixel"
        elif k in TERMS[a]:
            excluded[k] = "held"
        elif k in attacked:
            excluded[k] = "attacked"
        elif BACKBONE.get(k) in bb:
            excluded[k] = "backbone of the attacked metric"
        elif strict and WFAM[k] in {WFAM[t] for t in TERMS[a] if t in WFAM}:
            excluded[k] = "family of a held metric (strict)"
        else:
            wit.append(k)
    fams = sorted({WFAM[k] for k in wit if worse[k] > thr[k]})
    if not held:
        v = "not_held"
    elif not moved:
        v = "cannot_push"
    elif len(fams) >= 2:
        v = "real"
    else:
        v = "suspect"
    out.update(verdict=v, held=held, moved=moved, n_witness_families=len(fams), witness_families=",".join(fams), witnesses=",".join(wit),
               held_ratio=held_ratio, worse=worse, excluded=excluded)
    return out


def analyse():
    os.makedirs(OUT, exist_ok=True)
    T = pd.read_csv(P.THRESHOLDS).set_index("metric")
    thr, thr_alt, sm = T.p90.to_dict(), T.p90_alt_fill.to_dict(), T.s_m.to_dict()
    meta, means, cmean, views, ctrl_views = load()
    if REPRO:
        rp = pd.concat(REPRO)
        rp["mean_diff_over_p90"] = rp.mean_diff.abs() / rp.metric.map(thr)
        rp.to_csv(os.path.join(OUT, "repro_check.csv"), index=False)
        print("item 9 runs, stored vs recomputed scene means: largest |difference| / threshold without VIF =",
              round(float(rp[rp.metric != "vif"].mean_diff_over_p90.max()), 5), "; VIF (stored float32 vs recomputed float64): largest =",
              round(float(rp[rp.metric == "vif"].mean_diff_over_p90.max()), 3), "median =", round(float(rp[rp.metric == "vif"].mean_diff_over_p90.median()), 3))
    ok = meta[meta.status == "ok"]
    rows, long = [], []
    for row in ok[ok.gamma > 0].itertuples():
        z = ok[(ok.scene == row.scene) & (ok.anchor == row.anchor) & (ok.gamma == 0) & (ok.seed == 0)]
        zr = z.iloc[0].run
        c = classify(row, means, cmean, thr, zr)
        cs = classify(row, means, cmean, thr, zr, strict=True)
        ca = classify(row, means, cmean, thr_alt, zr)
        cz = classify(row, means, cmean, thr, zr, held_ref="zero")
        r = dict(run=row.run, scene=row.scene, anchor=row.anchor, target=row.target, gamma=row.gamma, seed=row.seed, source=row.source, zero_run=zr,
                 verdict=c["verdict"], verdict_strict=cs["verdict"], verdict_alt_thresholds=ca["verdict"], verdict_held_vs_gamma0=cz["verdict"],
                 held_vs_gamma0_worst=max(cz["held_ratio"].values()), held=c["held"], moved=c["moved"],
                 n_witness_families=c["n_witness_families"], witness_families=c["witness_families"], n_witness_families_strict=cs["n_witness_families"],
                 witnesses=c["witnesses"])
        cur, zero, con = means[row.run]["test"]["full"], means[zr]["test"]["full"], cmean[row.scene]["full"]
        for k in HELD[row.anchor]:
            r[f"held_{k}_vs_control"] = cur[k] - con[k]
            r[f"held_{k}_worse_over_p90"] = c["held_ratio"][k]
        for k in IMG:
            r[f"d_{k}"] = cur[k] - zero[k]
            r[f"z_{k}"] = SIGN[k] * (cur[k] - zero[k]) / sm[k]
            r[f"worse_over_p90_{k}"] = c["worse"][k] / thr[k]
            role = "witness" if k in c["witnesses"].split(",") else c["excluded"][k]
            long.append(dict(run=row.run, scene=row.scene, anchor=row.anchor, target=row.target, gamma=row.gamma, seed=row.seed, metric=k, role=role,
                             family=WFAM.get(k, "pixel"), control=con[k], gamma0=zero[k], value=cur[k], d_vs_control=cur[k] - con[k],
                             d_vs_gamma0=cur[k] - zero[k], z_vs_gamma0=r[f"z_{k}"], worse_over_p90=r[f"worse_over_p90_{k}"], p90=thr[k]))
        has_half = (row.run, "test", "half", "psnr") in means.index and (zr, "test", "half", "psnr") in means.index
        has_train = any((row.run, "train", q, "psnr") in means.index for q in ("stored", "full")) and any((zr, "train", q, "psnr") in means.index for q in ("stored", "full"))
        for k in c["attacked"].split(","):
            if c["worse"][k] < thr[k]:
                continue
            if has_half:
                rel_f = (cur[k] - zero[k]) / zero[k]
                rel_h = (means[row.run]["test"]["half"][k] - means[zr]["test"]["half"][k]) / means[zr]["test"]["half"][k]
                r[f"R_{k}"], r[f"rel_full_{k}"], r[f"rel_half_{k}"] = rel_h / rel_f, rel_f, rel_h
            if has_train:
                fr = "stored" if (row.run, "train", "stored", k) in means.index else "full"
                fz = "stored" if (zr, "train", "stored", k) in means.index else "full"
                dte = means[row.run]["test"][fr][k] - means[zr]["test"][fz][k]
                dtr = means[row.run]["train"][fr][k] - means[zr]["train"][fz][k]
                r[f"G_{k}"], r[f"d_train_{k}"] = dte / dtr, dtr
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "verdicts.csv"), index=False)
    pd.DataFrame(long).to_csv(os.path.join(OUT, "effects_long.csv"), index=False)
    zrows = []
    for row in ok[ok.gamma == 0].itertuples():
        cur, con = means[row.run]["test"]["full"], cmean[row.scene]["full"]
        zrows.append(dict(run=row.run, scene=row.scene, anchor=row.anchor, seed=row.seed,
                          **{f"d_{k}": cur[k] - con[k] for k in IMG}, **{f"z_{k}": SIGN[k] * (cur[k] - con[k]) / sm[k] for k in IMG},
                          **{f"worse_over_p90_{k}": -SIGN[k] * (cur[k] - con[k]) / thr[k] for k in IMG}))
    zres = pd.DataFrame(zrows)
    zres.to_csv(os.path.join(OUT, "gamma0_vs_control.csv"), index=False)
    return meta, res, zres, means, cmean, thr, sm


def controls(cmean, thr, sm):
    rows = []
    for scene in SCENE_NAME:
        f = os.path.join(P.S2_AUX, f"controls_{scene.replace('/', '_')}.parquet")
        if not os.path.exists(f):
            continue
        c = pd.read_parquet(f).groupby(["name", "res", "metric"]).value.mean()
        con = cmean[scene]
        for name in c.index.get_level_values(0).unique():
            r = dict(scene=scene, condition=name, d_psnr=c[name]["full"]["psnr"] - con["full"]["psnr"])
            for k in ("dists", "dreamsim", "flip"):
                d_f = c[name]["full"][k] - con["full"][k]
                r[f"worse_over_p90_{k}"] = d_f / thr[k]
                r[f"z_{k}"] = -d_f / sm[k]
                if d_f >= thr[k]:
                    r[f"R_{k}"] = ((c[name]["half"][k] - con["half"][k]) / con["half"][k]) / (d_f / con["full"][k])
            for k in ("cvvdp_image", "fsim", "vif", "lpips_alex", "lpips_vgg", "ssim", "ms_ssim"):
                r[f"worse_over_p90_{k}"] = -SIGN[k] * (c[name]["full"][k] - con["full"][k]) / thr[k]
            rows.append(r)
    if not rows:
        return None
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "controls.csv"), index=False)
    return res


def fmt(v, nd=1, plus=True):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}")


def point_table(res, title):
    L = [f"**{title}**", "",
         "| 场景 | 守住的 | 目标 | γ | 种子 | 守住的指标相对对照模型：变差量 ÷ 阈值（负 = 变好） | 被攻击指标（z；变差量 ÷ 阈值） | " +
         " | ".join("ColorVideoVDP (JOD)" if k == "cvvdp_image" else LABEL[k] for k in WCOLS) + " | 超阈值的见证族数 | 判定 | 严格口径 | 若以 γ = 0 的运行为基准判「守住」（未登记） | R | G |",
         "|:--|:--|:--|--:|--:|:--|:--|" + "--:|" * len(WCOLS) + "--:|:--|:--|:--|--:|--:|"]
    for r in res.itertuples():
        d = r._asdict()
        att = MULTI if r.target == "multi" else [r.target]
        wit = set(str(r.witnesses).split(","))
        held = "；".join(f"{LABEL[k]} {d[f'held_{k}_worse_over_p90']:+.1f}" for k in HELD[r.anchor])
        atk = "；".join(f"{LABEL[k]} {d[f'z_{k}']:+.1f}（{d[f'worse_over_p90_{k}']:.1f}）" for k in att)
        cells = []
        for k in WCOLS:
            v = f"{d[f'd_{k}']:+.2f}" if k == "cvvdp_image" else f"{d[f'z_{k}']:+.1f}"
            if k not in wit:
                v = f"（{v}）"
            elif d[f"worse_over_p90_{k}"] > 1:
                v = f"**{v}**"
            cells.append(v)
        R = "；".join(f"{d.get(f'R_{k}'):.2f}" for k in att if isinstance(d.get(f"R_{k}"), float) and not np.isnan(d.get(f"R_{k}")))
        G = "；".join(f"{d.get(f'G_{k}'):.2f}" for k in att if isinstance(d.get(f"G_{k}"), float) and not np.isnan(d.get(f"G_{k}")))
        L.append(f"| {SCENE_NAME[r.scene]} | {ANCHOR_NAME[r.anchor]} | {LABEL[r.target]} | {r.gamma:g} | {r.seed} | {held} | {atk} | " + " | ".join(cells) +
                 f" | {r.n_witness_families} | {VERDICT[r.verdict]} | {VERDICT[r.verdict_strict]} | {VERDICT[r.verdict_held_vs_gamma0]}（{r.held_vs_gamma0_worst:+.1f}） | {R or '—'} | {G or '—'} |")
    L += ["", "（见证指标各列：相对同一守住集合、γ = 0 的运行，z < 0 = 变差，ColorVideoVDP 为 JOD；**粗体** = 变差超过阈值并计入的见证；"
          "（括号）= 不是独立见证：守住的、被攻击的、或与被攻击指标同骨干。倒数第三列是未登记的旁证：把「守住了吗」的基准从原始对照模型换成同一守住集合、γ = 0 的运行时的判定，括号里是守住的指标里变差最多的那个 ÷ 阈值。R、G 只在被攻击指标变差超过阈值时给出；多目标的点按 DISTS；FLIP；DreamSim；VIF 的顺序列出其中超过阈值的。）", ""]
    return L


def matrix(res):
    m = res[(res.gamma == 0.05) & (res.seed == 0) & (res.target != "multi")]
    L = ["**充分性矩阵（γ = 0.05，种子 0；每格：三个场景各自的判定 → 结论）**", "",
         "| 守住的（= 论文里报告的） | 攻击 DISTS | 攻击 DreamSim | 攻击 FLIP |", "|:--|:--|:--|:--|"]
    concl = []
    for a in ("psnr", "default", "triplet"):
        cells = []
        for t in ("dists", "dreamsim", "flip"):
            g = m[(m.anchor == a) & (m.target == t)].set_index("scene")
            if not len(g):
                cells.append("（未做）")
                continue
            per = "；".join(f"{SCENE_NAME[s]}：{VERDICT[g.loc[s].verdict]}" for s in SCENE_NAME if s in g.index)
            if len(g) == 3 and g.verdict.nunique() == 1:
                c = f"**{VERDICT[g.verdict.iloc[0]]}（3/3）**"
            elif len(g) == 3:
                c = "**场景依赖**"
            else:
                c = f"只有 {len(g)} 个场景，不下结论"
            cells.append(f"{per} → {c}")
            concl.append(dict(anchor=a, target=t, n_scenes=len(g), conclusion=c.strip("*"), per_scene=per))
        L.append(f"| {ANCHOR_NAME[a]} | " + " | ".join(cells) + " |")
    pd.DataFrame(concl).to_csv(os.path.join(OUT, "matrix.csv"), index=False)
    L += ["", "**矩阵各格里见证指标的变化（相对 γ = 0 的运行；ColorVideoVDP 用 JOD，其余用 z；负 = 变差）**", "",
          "| 守住的 | 目标 | 场景 | 判定 | 被攻击指标 z | ColorVideoVDP | FSIM | VIF | FLIP | DreamSim | LPIPS-Alex | DISTS | MS-SSIM | R | G |",
          "|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for a in ("psnr", "default", "triplet"):
        for t in ("dists", "dreamsim", "flip"):
            for r in m[(m.anchor == a) & (m.target == t)].itertuples():
                d = r._asdict()
                wit = set(str(r.witnesses).split(","))
                def cell(k):
                    v = f"{d[f'd_{k}']:+.2f} JOD" if k == "cvvdp_image" else f"{d[f'z_{k}']:+.1f}"
                    return f"（{v}）" if k not in wit else (f"**{v}**" if d[f"worse_over_p90_{k}"] > 1 else v)
                Rv, Gv = d.get(f"R_{t}"), d.get(f"G_{t}")
                L.append(f"| {ANCHOR_NAME[a]} | {LABEL[t]} | {SCENE_NAME[r.scene]} | {VERDICT[r.verdict]} | {d[f'z_{t}']:+.1f} | " +
                         " | ".join(cell(k) for k in ("cvvdp_image", "fsim", "vif", "flip", "dreamsim", "lpips_alex", "dists", "ms_ssim")) +
                         f" | {fmt(Rv, 2, False) or '—'} | {fmt(Gv, 2, False) or '—'} |")
    return L + [""]


def seeds_table(res, thr, sm):
    L, rows = [], []
    key = ["scene", "anchor", "target", "gamma"]
    for k, g in res.groupby(key):
        if g.seed.nunique() < 2:
            continue
        a, b = g[g.seed == 0].iloc[0], g[g.seed == 1].iloc[0]
        row = dict(zip(key, k), verdict_s0=a.verdict, verdict_s1=b.verdict)
        for m in IMG:
            row[f"z_s0_{m}"], row[f"z_s1_{m}"] = a[f"z_{m}"], b[f"z_{m}"]
            row[f"absdiff_over_p90_{m}"] = abs(a[f"d_{m}"] - b[f"d_{m}"]) / thr[m]
        for m in (MULTI if k[2] == "multi" else [k[2]]):
            for q in ("R", "G"):
                row[f"{q}_s0"], row[f"{q}_s1"] = a.get(f"{q}_{m}"), b.get(f"{q}_{m}")
        rows.append(row)
    if not rows:
        return L
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "seeds.csv"), index=False)
    show = ["psnr", "ssim", "lpips_vgg", "dists", "dreamsim", "flip", "lpips_alex", "cvvdp_image", "fsim", "vif"]
    L += ["**D 组：同一配置的两个微调种子（相对种子 0 的 γ = 0 运行；每格 = 种子 0 / 种子 1 的 z；括号 = 两个种子的差 ÷ 阈值）**", "",
          "| 场景 | 守住的 | 目标 | 判定（种子 0 / 种子 1） | " + " | ".join(LABEL[m] for m in show) + " | R | G |", "|:--|:--|:--|:--|" + "--:|" * (len(show) + 2)]
    for r in s.itertuples():
        d = r._asdict()
        L.append(f"| {SCENE_NAME[r.scene]} | {ANCHOR_NAME[r.anchor]} | {LABEL[r.target]} | {VERDICT[r.verdict_s0]} / {VERDICT[r.verdict_s1]} | " +
                 " | ".join(f"{d[f'z_s0_{m}']:+.1f} / {d[f'z_s1_{m}']:+.1f}（{d[f'absdiff_over_p90_{m}']:.2f}）" for m in show) +
                 f" | {fmt(d['R_s0'], 2, False)} / {fmt(d['R_s1'], 2, False)} | {fmt(d['G_s0'], 2, False)} / {fmt(d['G_s1'], 2, False)} |")
    return L + [""]


def groups():
    g = {}
    for line in open(os.path.join(HERE, "jobs.txt")):
        p = line.split()
        if len(p) == 4 and not line.startswith("#"):
            _, a, t, gm = p[2].split(":")
            g[(p[1], a, t, float(gm), int(p[3]))] = p[0]
    return g


def main():
    meta, res, zres, means, cmean, thr, sm = analyse()
    tabs = {}
    bad = meta[meta.status != "ok"]
    v = res.copy()
    order = {"psnr": 0, "default": 1, "triplet": 2}
    v["_o"] = v.anchor.map(order)
    g = groups()
    v["group"] = [("A" if r.source == "item9" else g.get((r.scene, r.anchor, r.target, r.gamma, r.seed), "?")) for r in v.itertuples()]
    v = v.sort_values(["scene", "_o", "target", "gamma", "seed"])
    v.drop(columns="_o").to_csv(os.path.join(OUT, "verdicts.csv"), index=False)
    tabs["matrix"] = matrix(v)
    x = v[(v.group == "X") & (v.seed == 0)]
    if len(x):
        per = "；".join(f"{SCENE_NAME[r.scene]}：{VERDICT[r.verdict]}" for r in x.sort_values("scene", key=lambda c: c.map(list(SCENE_NAME).index)).itertuples())
        if len(x) == 3 and x.verdict.nunique() == 1:
            c = f"**{VERDICT[x.verdict.iloc[0]]}（3/3）**"
        elif len(x) == 3:
            c = "**场景依赖**"
        else:
            c = f"只有 {len(x)} 个场景，不下结论"
        tabs["Xsummary"] = [f"**X 组的三场景判定（三个都守、只攻击 DreamSim、γ = 0.0125）**：{per} → {c}", ""]
    tabs["A"] = point_table(v[v.group == "A"], "A 组：第 9 条已有的 truck 运行按登记规则重新判定")
    for grp, title in (("B", "B 组：truck 上的单目标攻击"), ("C", "C 组：bonsai 与 garden"), ("D", "D 组：truck，第二个微调种子"), ("E", "E 组：有余时做的点"),
                       ("X", "X 组（登记清单之外，见 prereg_addendum_1.md）：三个都守、只攻击 DreamSim、γ = 0.0125"),
                       ("Y", "Y 组（登记清单之外，见 prereg_addendum_2.md）：bonsai、garden 上更弱的推力")):
        if (v.group == grp).any():
            tabs[grp] = point_table(v[v.group == grp], title)
    lad = v[(v.anchor == "triplet") & (v.target == "dreamsim") & (v.seed == 0)]
    if lad.gamma.nunique() > 1:
        L = ["**三个都守、只攻击 DreamSim：推力梯子上各点的判定**（每格：判定；DreamSim 的 z；超过阈值的见证族数；守住的指标里变差最多的 ÷ 阈值）", "",
             "| 场景 | " + " | ".join(f"γ = {g:g}" for g in sorted(lad.gamma.unique(), reverse=True)) + " | 守住了的最大 γ 的判定 |", "|:--|" + ":--|" * (lad.gamma.nunique() + 1)]
        finals = []
        for sc in SCENE_NAME:
            cells, final = [], "该场景没有守住的点"
            for gm in sorted(lad.gamma.unique(), reverse=True):
                r = lad[(lad.scene == sc) & (lad.gamma == gm)]
                if not len(r):
                    cells.append("（未做）")
                    continue
                r = r.iloc[0]
                worst = max(r[f"held_{k}_worse_over_p90"] for k in HELD["triplet"])
                cells.append(f"{VERDICT[r.verdict]}；{r.z_dreamsim:+.1f}；{r.n_witness_families}；{worst:+.1f}")
                if r.held and final == "该场景没有守住的点":
                    final = f"{VERDICT[r.verdict]}（γ = {gm:g}）"
            finals.append(final.split("（")[0])
            L.append(f"| {SCENE_NAME[sc]} | " + " | ".join(cells) + f" | {final} |")
        concl = f"**{finals[0]}（3/3）**" if len(set(finals)) == 1 else "**场景依赖**"
        L += ["", f"三个场景合起来：{concl}", ""]
        tabs["ladder"] = L
        pd.DataFrame(dict(scene=list(SCENE_NAME), verdict_at_largest_held_gamma=finals)).to_csv(os.path.join(OUT, "ladder.csv"), index=False)
    tabs["seeds"] = seeds_table(v, thr, sm)
    if len(zres):
        L = ["**γ = 0 的运行相对原始对照模型（守住集合本身带来的变化；z，正 = 变好；ColorVideoVDP 为 JOD）**", "",
             "| 场景 | 守住的 | ΔPSNR (dB) | ΔSSIM | ΔLPIPS-VGG | DISTS | DreamSim | LPIPS-Alex | FLIP | ColorVideoVDP | FSIM | VIF |", "|:--|:--|" + "--:|" * 10]
        zz = zres.assign(_o=zres.anchor.map(order)).sort_values(["scene", "_o"])
        for r in zz.itertuples():
            d = r._asdict()
            L.append(f"| {SCENE_NAME[r.scene]} | {ANCHOR_NAME[r.anchor]} | {d['d_psnr']:+.2f} | {d['d_ssim']:+.3f} | {d['d_lpips_vgg']:+.3f} | " +
                     " | ".join(f"{d[f'z_{k}']:+.1f}" for k in ("dists", "dreamsim", "lpips_alex", "flip")) + f" | {d['d_cvvdp_image']:+.2f} JOD | {d['z_fsim']:+.1f} | {d['z_vif']:+.1f} |")
        tabs["gamma0"] = L + [""]
    c = controls(cmean, thr, sm)
    if c is not None:
        name = {"blur15": "真实退化：高斯模糊 σ = 1.5 px", "tint": "真实退化：低频色偏 ±6/255", "adv_dists": "二维对抗样本：攻击 DISTS（±2/255）",
                "adv_dreamsim": "二维对抗样本：攻击 DreamSim（±2/255）", "adv_flip": "二维对抗样本：攻击 FLIP（±2/255）"}
        L = ["**读数 R 的校准（对照模型的 20 张 held-out 渲染图；相对未处理的渲染图；truck 是登记的校准，bonsai、garden 是未登记的同参数扩展）**", "",
             "| 场景 | 处理 | ΔPSNR (dB) | DISTS：z（R） | DreamSim：z（R） | FLIP：z（R） | 其余指标变差量 ÷ 阈值：ColorVideoVDP / FSIM / VIF / LPIPS-Alex / LPIPS-VGG / SSIM / MS-SSIM |", "|:--|:--|--:|--:|--:|--:|:--|"]
        for r in c.itertuples():
            d = r._asdict()
            cells = [f"{d[f'z_{k}']:+.1f}（{fmt(d.get(f'R_{k}'), 2, False) or '变化不到阈值'}）" for k in ("dists", "dreamsim", "flip")]
            oth = " / ".join(f"{d[f'worse_over_p90_{k}']:+.1f}" for k in ("cvvdp_image", "fsim", "vif", "lpips_alex", "lpips_vgg", "ssim", "ms_ssim"))
            L.append(f"| {SCENE_NAME[r.scene]} | {name[r.condition]} | {r.d_psnr:+.2f} | " + " | ".join(cells) + f" | {oth} |")
        tabs["controls"] = L + [""]
    L = []
    if len(bad):
        L += ["**发散或失败、不计入判定的运行**：" + "；".join(f"{r.run}（{r.status}）" for r in bad.itertuples()), ""]
    t = meta[(meta.source == "new") & (meta.status == "ok")]
    if len(t):
        L += [f"新运行 {len(t)} 次：训练用时合计 {t.train_time.sum() / 60:.0f} 分钟，训练加评测的进程用时合计 {t.wall_time.sum() / 60:.0f} 分钟（并行运行，不等于墙上时间）。", ""]
    tabs["runs"] = L
    diff = v[(v.verdict != v.verdict_strict) | (v.verdict != v.verdict_alt_thresholds)]
    L = ["**换一种口径会改变判定的点**（严格口径：再排除与守住的指标同族的见证；另一种阈值补法：见 thresholds.csv）：" +
         ("没有。" if not len(diff) else "")]
    for r in diff.itertuples():
        L.append(f"- {SCENE_NAME[r.scene]}，{ANCHOR_NAME[r.anchor]}，攻击 {LABEL[r.target]}，γ = {r.gamma:g}，种子 {r.seed}：主判定 {VERDICT[r.verdict]}；严格口径 {VERDICT[r.verdict_strict]}；"
                 f"另一种阈值补法 {VERDICT[r.verdict_alt_thresholds]}")
    tabs["sensitivity"] = L + [""]
    for k, lines in tabs.items():
        open(os.path.join(OUT, f"table_{k}.md"), "w").write("\n".join(lines) + "\n")
    allt = [x for k in tabs for x in tabs[k]]
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(allt) + "\n")
    print("\n".join(allt))
if __name__ == "__main__":
    main()
