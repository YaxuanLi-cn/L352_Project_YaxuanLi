#!/usr/bin/env python
import glob
import os
import sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as PATHS              
OUT, RES = os.path.join(PATHS.DECOMPOSITION, "out"), os.path.join(PATHS.DECOMPOSITION, "results")
CTRL = "default__ctrl"
NB = 2000
ORDER = ["l1__pure", "l2__pure", "dssim__pure", "msssim__pure", "flip__pure", "cvvdp__pure", "lpips_vgg__pure",
         "lpips_alex__pure", "dists__pure", "l2__mixed", "msssim__mixed", "flip__mixed", "cvvdp__mixed",
         "lpips_vgg__mixed", "lpips_alex__mixed", "dists__mixed"]
DEEP = ["lpips_vgg__pure", "lpips_alex__pure", "dists__pure", "lpips_vgg__mixed", "lpips_alex__mixed", "dists__mixed"]


def psnr(x):
    return -10 * np.log10(np.maximum(x, 1e-12))


def boot(fn, n_scene, rng):
    idx = rng.integers(0, n_scene, size=(NB, n_scene))
    vals = np.array([fn(i) for i in idx])
    return np.nanpercentile(vals, [2.5, 97.5], axis=0)


def fmt(v, lo, hi, d=2):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def step1(lines):
    px = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "pixel_*.csv")))])
    px["single"] = 0.5 * (px.mse_AG + px.mse_BG)
    px["v"] = 0.5 * px.mse_AB
    px["bias2"] = px.single - px.v
    ident = (px.mse_MG - (px.single - px.mse_AB / 4)).abs() / px.single
    asym = ((px.mse_AG - px.mse_BG).abs() / px.single)
    px["psnr_single"] = 0.5 * (psnr(px.mse_AG) + psnr(px.mse_BG))
    px["psnr_mean"] = psnr(px.mse_MG)
    px["psnr_between"] = psnr(px.mse_AB)
    px["psnr_bias"] = psnr(px.bias2)
    g = px.groupby(["scene", "model"], sort=False)[["single", "v", "bias2", "mse_MG", "mse_AB", "psnr_single", "psnr_mean",
                                                    "psnr_between", "psnr_bias"]].mean().reset_index()
    scenes = sorted(g.scene.unique())
    lines.append(f"场景数 {len(scenes)}，视角数 {px.groupby('scene').view.nunique().to_dict()}")
    lines.append(f"代数恒等式 mse(M,G) = 单次平均 − mse(A,B)/4 的最大相对误差：{ident.max():.2e}（它对任意两张图恒成立，只是代码检查，"
                 f"不能验证独立性假设）")
    lines.append(f"两次训练各自到参考图的 MSE 之差 |mse(A,G) − mse(B,G)| / 平均：中位数 {asym.median():.4f}，95 分位 {asym.quantile(0.95):.4f}"
                 f"（两次训练的地位对称）")
    neg = (px.bias2 <= 0).mean()
    lines.append(f"逐视角 bias² ≤ 0 的比例：{neg:.4f}")
    c = g[g.model == CTRL].set_index("scene")
    rows, per_scene = [], []
    rng = np.random.default_rng(0)
    for model in [CTRL] + ORDER:
        m = g[g.model == model].set_index("scene").loc[scenes]
        cc = c.loc[scenes]
        e = ((m.single - cc.single) / cc.single).to_numpy()
        r = ((m.v - cc.v) / cc.single).to_numpy()
        bb = ((m.bias2 - cc.bias2) / cc.single).to_numpy()
        vshare = (m.v / m.single).to_numpy()
        d_single = (m.psnr_single - cc.psnr_single).to_numpy()
        d_mean = (m.psnr_mean - cc.psnr_mean).to_numpy()
        d_bias = (m.psnr_bias - cc.psnr_bias).to_numpy()
        gain = (m.psnr_mean - m.psnr_single).to_numpy()
        between = m.psnr_between.to_numpy()
        for i, s in enumerate(scenes):
            per_scene.append(dict(scene=s, model=model, mse_single=m.single.iloc[i], v=m.v.iloc[i], bias2=m.bias2.iloc[i],
                                  excess_rel=e[i], excess_v_rel=r[i], excess_bias_rel=bb[i],
                                  irreproducible_share=(r[i] / e[i]) if abs(e[i]) > 0.02 else np.nan,
                                  v_share_of_error=vshare[i], psnr_single=m.psnr_single.iloc[i], psnr_mean_image=m.psnr_mean.iloc[i],
                                  psnr_between_runs=between[i], psnr_bias_only=m.psnr_bias.iloc[i],
                                  dpsnr_single=d_single[i], dpsnr_mean_image=d_mean[i], dpsnr_bias_only=d_bias[i]))
        row = dict(model=model)
        for name, arr in (("excess_rel", e), ("v_share_of_error", vshare), ("dpsnr_single", d_single), ("dpsnr_mean_image", d_mean),
                          ("dpsnr_bias_only", d_bias), ("averaging_gain_db", gain), ("psnr_between_runs", between)):
            lo, hi = boot(lambda i, a=arr: a[i].mean(), len(scenes), rng)
            row[name], row[name + "_lo"], row[name + "_hi"] = arr.mean(), lo, hi
        if model != CTRL:
            share = r.mean() / e.mean()
            lo, hi = boot(lambda i: r[i].mean() / e[i].mean(), len(scenes), rng)
            row.update(irreproducible_share=share, irreproducible_share_lo=lo, irreproducible_share_hi=hi,
                       excess_reliable=bool(e.mean() > 0.02 and (e > 0).sum() >= len(scenes) - 1),
                       n_scenes_excess_positive=int((e > 0).sum()),
                       n_scenes_share_above_half=int(np.sum((r / np.where(np.abs(e) > 0.02, e, np.nan)) > 0.5)))
        rows.append(row)
    os.makedirs(RES, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(RES, "decomposition_per_loss.csv"), index=False)
    pd.DataFrame(per_scene).to_csv(os.path.join(RES, "decomposition_per_scene.csv"), index=False)
    lines.append("")
    lines.append("| 模型 | 单次训练相对对照的 MSE 增量（对照 MSE 的倍数） | 其中不可复现的份额 | 模型自身误差里不可复现的份额 | ΔPSNR 单次 (dB) | ΔPSNR 两次平均图 (dB) | "
                 "ΔPSNR 只留可复现部分 (dB) | 两次训练之间的 PSNR (dB) |")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for _, r_ in per.iterrows():
        if r_.model == CTRL:
            share = "—"
        elif not r_.excess_reliable:
            share = "（增量过小或符号不定，不计算）"
        else:
            share = fmt(r_.irreproducible_share, r_.irreproducible_share_lo, r_.irreproducible_share_hi)
        lines.append(f"| {r_.model} | {fmt(r_.excess_rel, r_.excess_rel_lo, r_.excess_rel_hi)} | {share} | "
                     f"{fmt(r_.v_share_of_error, r_.v_share_of_error_lo, r_.v_share_of_error_hi, 3)} | "
                     f"{fmt(r_.dpsnr_single, r_.dpsnr_single_lo, r_.dpsnr_single_hi)} | "
                     f"{fmt(r_.dpsnr_mean_image, r_.dpsnr_mean_image_lo, r_.dpsnr_mean_image_hi)} | "
                     f"{fmt(r_.dpsnr_bias_only, r_.dpsnr_bias_only_lo, r_.dpsnr_bias_only_hi)} | "
                     f"{fmt(r_.psnr_between_runs, r_.psnr_between_runs_lo, r_.psnr_between_runs_hi, 1)} |")
    lines.append("")
    lines.append("ΔPSNR 都是相对对照模型的同类量（单次对单次、平均图对平均图、可复现部分对可复现部分），按场景等权平均；方括号为按场景自助法 95% 区间。")
    lines.append("")
    lines.append("逐场景（深度特征损失）：不可复现份额（括号内：MSE 增量 ÷ 对照 MSE；增量不足 2% 时份额记为 —），以及两次训练之间的 PSNR（dB）")
    lines.append("")
    psd = pd.DataFrame(per_scene)
    lines.append("| 场景 | " + " | ".join(DEEP) + " |")
    lines.append("|:--|" + "--:|" * len(DEEP))
    for s in scenes:
        cells = []
        for model in DEEP:
            q = psd[(psd.scene == s) & (psd.model == model)].iloc[0]
            sh = "—" if not np.isfinite(q.irreproducible_share) else f"{q.irreproducible_share:.2f}"
            cells.append(f"{sh}（{q.excess_rel:+.2f}）, {q.psnr_between_runs:.1f}")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return per


def step3(lines):
    files = sorted(glob.glob(os.path.join(OUT, "tiles_*.csv")))
    t = pd.concat([pd.read_csv(f) for f in files])
    cols = ["pearson_excess", "spearman_excess", "pearson_excess_mean_image", "spearman_excess_mean_image", "pearson_plain_error"]
    g = t.groupby(["scene", "model"], sort=False)[cols].mean().reset_index()
    scenes = sorted(g.scene.unique())
    rng = np.random.default_rng(1)
    rows = []
    for model in ORDER:
        m = g[g.model == model].set_index("scene").loc[scenes]
        row = dict(model=model)
        for c in cols:
            a = m[c].to_numpy()
            lo, hi = boot(lambda i, a=a: a[i].mean(), len(scenes), rng)
            row[c], row[c + "_lo"], row[c + "_hi"] = a.mean(), lo, hi
        row["n_scenes_positive"] = int((m.pearson_excess > 0).sum())
        rows.append(row)
    loc = pd.DataFrame(rows)
    loc.to_csv(os.path.join(RES, "localisation_per_loss.csv"), index=False)
    g.to_csv(os.path.join(RES, "localisation_per_scene.csv"), index=False)
    lines.append("")
    lines.append("| 模型 | 两次训练之差 与 超出对照的误差（单次） Pearson | Spearman | 与 平均图超出对照的误差 Pearson | 与 单次误差本身 Pearson | 为正的场景数 |")
    lines.append("|:--|--:|--:|--:|--:|--:|")
    for _, r_ in loc.iterrows():
        lines.append(f"| {r_.model} | {fmt(r_.pearson_excess, r_.pearson_excess_lo, r_.pearson_excess_hi)} | "
                     f"{fmt(r_.spearman_excess, r_.spearman_excess_lo, r_.spearman_excess_hi)} | "
                     f"{fmt(r_.pearson_excess_mean_image, r_.pearson_excess_mean_image_lo, r_.pearson_excess_mean_image_hi)} | "
                     f"{fmt(r_.pearson_plain_error, r_.pearson_plain_error_lo, r_.pearson_plain_error_hi)} | {r_.n_scenes_positive}/{len(scenes)} |")
    lines.append("")
    lines.append("16×16 像素块，逐视角在块上求相关，再对视角、场景平均；区间为按场景自助法。")


def step2(lines):
    from gh import metrics as M
    files = sorted(glob.glob(os.path.join(OUT, "metrics_*.csv")))
    scale = pd.read_csv(os.path.join(PATHS.ORIG_TABLES, "column_scale.csv")).set_index("metric").scale
    IMG = M.IMAGE_METRICS
    df = pd.concat([pd.read_csv(f) for f in files])
    tabs = None
    notes = []
    for scene in sorted(df.scene.unique()):
        d = df[df.scene == scene]
        if set(d.image.unique()) == {"M"}:
            if tabs is None:
                tabs = {"A": pd.read_csv(os.path.join(PATHS.ORIG_RESULTS, "metrics.csv")),
                        "B": pd.read_csv(os.path.join(PATHS.RERUN_RESULTS, "metrics.csv"))}
            views, mods = sorted(d.view.unique()), set(d.model.unique())
            for im, tb in tabs.items():
                q = tb[(tb.scene == scene) & (tb.seed == 0) & (tb.split == "test") & tb.metric.isin(IMG) & tb.view_id.isin(views)].copy()
                q["model"] = q.loss + "__" + q.variant
                q = q[q.model.isin(mods)]
                df = pd.concat([df, pd.DataFrame(dict(scene=scene, model=q.model, image=im, view=q.view_id, metric=q.metric, value=q.value))])
            pf = os.path.join(OUT, f"pixel_{scene.replace('/', '_')}.csv")
            px = pd.read_csv(pf)
            px = px[(px.model == CTRL) & px.view.isin(views)].sort_values("view")
            for im, col in (("A", "mse_AG"), ("B", "mse_BG")):
                t_ = df[(df.scene == scene) & (df.model == CTRL) & (df.image == im) & (df.metric == "psnr")].sort_values("view").value.to_numpy()
                notes.append(f"{scene} 表中 {im} 的 PSNR 与本分析像素表的最大差：{np.abs(t_ - psnr(px[col].to_numpy())).max():.2e} dB")
    piv = df.pivot_table(index=["scene", "model", "image", "view"], columns="metric", values="value")[IMG]
    sign = np.array([M.SIGN[k] for k in IMG])
    out, zrows = [], []
    for scene in sorted(df.scene.unique()):
        P = piv.loc[scene]
        c = {im: P.loc[(CTRL, im)].sort_index().to_numpy() for im in "ABM"}
        for model in ORDER:
            x = {im: P.loc[(model, im)].sort_index().to_numpy() for im in "ABM"}
            z_single = (0.5 * ((x["A"] - c["A"]) + (x["B"] - c["B"])) * sign).mean(0) / scale[IMG].to_numpy()
            z_mean = ((x["M"] - c["M"]) * sign).mean(0) / scale[IMG].to_numpy()
            z_mean_vs_single = ((x["M"] - 0.5 * (c["A"] + c["B"])) * sign).mean(0) / scale[IMG].to_numpy()
            loss = model.split("__")[0]
            j = M.LOSSES[loss]
            others = [k for k in IMG if M.FAMILY[k] != M.FAMILY[j]]
            oi = [IMG.index(k) for k in others]
            ji = IMG.index(j)
            for tag, z in (("single", z_single), ("mean_image", z_mean), ("mean_image_vs_single_ctrl", z_mean_vs_single)):
                out.append(dict(scene=scene, model=model, image=tag, z_own=z[ji], z_ref=z[oi].mean(), S=z[ji] - z[oi].mean()))
                for k, v in zip(IMG, z):
                    zrows.append(dict(scene=scene, model=model, image=tag, metric=k, z=v))
    o = pd.DataFrame(out)
    o.to_csv(os.path.join(RES, "mean_image_S_by_scene.csv"), index=False)
    pd.DataFrame(zrows).to_csv(os.path.join(RES, "mean_image_z_all_metrics.csv"), index=False)
    a = o.groupby(["model", "image"], sort=False)[["z_own", "z_ref", "S"]].mean().unstack("image")
    a.columns = [f"{c}_{i}" for c, i in a.columns]
    a = a.loc[ORDER].reset_index()
    a.to_csv(os.path.join(RES, "mean_image_S.csv"), index=False)
    nv = df.groupby("scene").view.nunique().to_dict()
    lines += notes
    lines.append("")
    lines.append(f"场景与视角数：{nv}。z = 相对对照的提升 ÷ s_m（论文的 column_scale.csv）；单次 = 两次训练各自相对各自对照的平均；"
                 f"平均图 = 两次训练的平均图相对对照的平均图。S = z_own − 其他族指标的平均 z（论文定义）。各场景等权平均。")
    lines.append("")
    lines.append("| 模型 | z_own 单次 | z_own 平均图 | z_ref 单次 | z_ref 平均图 | S 单次 | S 平均图 | S 平均图（对照用单次） |")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for _, r_ in a.iterrows():
        lines.append(f"| {r_.model} | {r_.z_own_single:+.2f} | {r_.z_own_mean_image:+.2f} | {r_.z_ref_single:+.2f} | {r_.z_ref_mean_image:+.2f} | "
                     f"{r_.S_single:+.2f} | {r_.S_mean_image:+.2f} | {r_.S_mean_image_vs_single_ctrl:+.2f} |")
    lines.append("")
    lines.append("逐场景（深度特征损失）：")
    lines.append("")
    lines.append("| 场景 | 模型 | z_own 单次 → 平均图 | z_ref 单次 → 平均图 | S 单次 → 平均图 |")
    lines.append("|:--|:--|--:|--:|--:|")
    w = o.pivot_table(index=["scene", "model"], columns="image", values=["z_own", "z_ref", "S"])
    for scene in sorted(o.scene.unique()):
        for model in DEEP:
            r_ = w.loc[(scene, model)]
            lines.append(f"| {scene} | {model} | {r_[('z_own', 'single')]:+.2f} → {r_[('z_own', 'mean_image')]:+.2f} | "
                         f"{r_[('z_ref', 'single')]:+.2f} → {r_[('z_ref', 'mean_image')]:+.2f} | "
                         f"{r_[('S', 'single')]:+.2f} → {r_[('S', 'mean_image')]:+.2f} |")


def main():
    lines = ["## 1. 像素域分解（8 个场景 × 17 个模型，试点的全部 held-out 视角）", ""]
    step1(lines)
    lines += ["", "## 3. 两次训练之差能否定位误差", ""]
    step3(lines)
    lines += ["", "## 2. 两次训练的平均图：13 个指标", ""]
    step2(lines)
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
if __name__ == "__main__":
    main()
