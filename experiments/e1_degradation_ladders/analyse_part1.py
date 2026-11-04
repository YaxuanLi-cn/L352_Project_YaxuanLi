import itertools
import os
import sys
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import numpy as np              
import pandas as pd              
from scipy import stats as sstats              
from nvsqa import stats              
from nvsqa.metrics import LABEL, SIGN              
IMG = ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg", "flip", "cvvdp_image"]
VID = ["cvvdp", "cvvdp_sust"]
CORE = ["psnr", "ssim", "lpips_vgg", "flip", "cvvdp_image"]
OUT = P.PART1_RESULTS


def load():
    cond = pd.read_csv(os.path.join(OUT, "conditions.csv"))
    cond = cond[cond.seed == 42].reset_index(drop=True)
    cond["scene_id"] = cond.dataset + "/" + cond.scene
    cond["dtype"] = cond.method + ":" + cond.ladder
    return cond


def seed_noise(metrics):
    c = pd.read_csv(os.path.join(OUT, "conditions.csv"))
    c["scene_id"] = c.dataset + "/" + c.scene
    main = c[c.seed == 42].set_index(["scene_id", "method", "condition"])
    alt = c[c.seed != 42].copy()
    alt["condition"] = alt.condition.str.replace(r"-s\d+$", "", regex=True)
    alt = alt.set_index(["scene_id", "method", "condition"])
    d = (alt[metrics] - main.loc[alt.index, metrics]).abs()
    d.reset_index().to_csv(os.path.join(OUT, "seed_noise.csv"), index=False)
    return d, d.quantile(0.9).to_dict()


def ladders(cond):
    for (scene, method), g in cond.groupby(["scene_id", "method"]):
        full = g[(g.ladder == "iterations") & (g.level == g[g.ladder == "iterations"].level.max())]
        for ladder, gl in g.groupby("ladder"):
            if ladder == "views":
                gl = pd.concat([gl, full.assign(level=full.num_train_views)])
            yield scene, method, ladder, gl.sort_values("level")


def monotonicity(cond, metrics):
    rows = []
    for scene, method, ladder, g in ladders(cond):
        if len(g) < 3:
            continue
        for m in metrics:
            if m not in g or g[m].isna().any():
                continue
            v = SIGN[m] * g[m].to_numpy(float)
            rho = sstats.spearmanr(g.level, v).statistic
            rows.append(dict(scene=scene, method=method, ladder=ladder, metric=m, n=len(g), spearman=rho,
                             inversions=int((np.diff(v) < 0).sum()), best_level=g.level.iloc[int(np.argmax(v))],
                             span=float(v.max() - v.min())))
    return pd.DataFrame(rows)


def fisher_mean(rs):
    rs = np.clip(np.asarray(rs, float), -0.9999, 0.9999)
    rs = rs[np.isfinite(rs)]
    return float(np.tanh(np.arctanh(rs).mean())) if len(rs) else np.nan


def intermetric(table, metrics, group):
    mats = []
    for _, g in table.groupby(group):
        if len(g) < 4:
            continue
        x = np.stack([SIGN[m] * g[m].to_numpy(float) for m in metrics], 1)
        mats.append(sstats.spearmanr(x).statistic)
    mats = np.stack(mats)
    mean = np.array([[fisher_mean(mats[:, i, j]) if i != j else 1.0 for j in range(len(metrics))]
                     for i in range(len(metrics))])
    lab = [LABEL[m] for m in metrics]
    return pd.DataFrame(mean, index=lab, columns=lab), pd.DataFrame(mats.min(0), index=lab, columns=lab)


def pair_kind(a, b):
    if a.method == b.method and a.ladder == b.ladder:
        return "same method, same ladder"
    if a.method == b.method:
        return "same method, different ladder"
    return "3DGS vs NeRF"


def disagreements(cond, metrics, thr=None):
    thr = thr or {}
    z = cond.copy()
    for m in metrics:
        z[m] = cond.groupby("scene_id")[m].transform(lambda v, s=SIGN[m]: (s * v - (s * v).mean()) / (s * v).std())
        z["raw_" + m] = SIGN[m] * cond[m]
    counts = {}
    cases = []
    for scene, g in z.groupby("scene_id"):
        recs = g.to_dict("records")
        for a, b in itertools.combinations(recs, 2):
            kind = pair_kind(pd.Series(a), pd.Series(b))
            for m1, m2 in itertools.combinations(metrics, 2):
                d1, d2 = a[m1] - b[m1], a[m2] - b[m2]
                if not (np.isfinite(d1) and np.isfinite(d2)) or d1 == 0 or d2 == 0:
                    continue
                r1, r2 = a["raw_" + m1] - b["raw_" + m1], a["raw_" + m2] - b["raw_" + m2]
                resolved = abs(r1) > thr.get(m1, 0.0) and abs(r2) > thr.get(m2, 0.0)
                key = (scene, m1, m2, kind)
                c = counts.setdefault(key, [0, 0, 0, 0])
                c[1] += 1
                c[3] += resolved
                if d1 * d2 < 0:
                    c[0] += 1
                    c[2] += resolved
                    cases.append(dict(scene=scene, metric_a=m1, metric_b=m2, kind=kind, resolved=resolved,
                                      cond_1=f"{a['method']}/{a['condition']}", cond_2=f"{b['method']}/{b['condition']}",
                                      dz_a=d1, dz_b=d2, d_a=SIGN[m1] * r1, d_b=SIGN[m2] * r2,
                                      strength=min(abs(d1), abs(d2))))
    rows = [dict(scene=k[0], metric_a=k[1], metric_b=k[2], kind=k[3], disagree=v[0], pairs=v[1], robust=v[2],
                 resolved_pairs=v[3]) for k, v in counts.items()]
    by_scene = pd.DataFrame(rows)
    num = ["disagree", "pairs", "robust", "resolved_pairs"]
    by_type = by_scene.groupby(["metric_a", "metric_b", "kind"])[num].sum().reset_index()
    total = by_scene.groupby(["metric_a", "metric_b"])[num].sum().reset_index()
    for t in (total, by_type, by_scene):
        t["rate"] = t.disagree / t.pairs
        t["robust_rate"] = t.robust / t.pairs
    by_scene.to_csv(os.path.join(OUT, "disagreement_by_scene.csv"), index=False)
    cases = pd.DataFrame(cases).sort_values("strength", ascending=False)
    return total, by_type, cases


def matched_psnr(cond, metrics, tol=0.3):
    rows = []
    for scene, g in cond.groupby("scene_id"):
        a, b = g[g.method == "3dgs"], g[g.method == "nerf"]
        for _, n in b.iterrows():
            cand = a.iloc[(a.psnr - n.psnr).abs().argsort()]
            if abs(cand.iloc[0].psnr - n.psnr) < tol:
                g3 = cand.iloc[0]
                row = dict(scene=scene, nerf=n.condition, gs=g3.condition, psnr_nerf=n.psnr, psnr_3dgs=g3.psnr)
                for m in metrics:
                    row["d_" + m] = SIGN[m] * (g3[m] - n[m])
                    row["dz_" + m] = row["d_" + m] / (SIGN[m] * g[m]).std()
                rows.append(row)
    return pd.DataFrame(rows)


def type_bias(cond, metrics):
    z = pd.DataFrame({m: cond.groupby("scene_id")[m].transform(
        lambda v, s=SIGN[m]: (s * v - (s * v).mean()) / (s * v).std()) for m in metrics})
    consensus = z.mean(axis=1)
    dev = z.sub(consensus, axis=0)
    dev["dtype"] = cond.dtype.to_numpy()
    dev["scene_id"] = cond.scene_id.to_numpy()
    per_scene = dev.groupby(["dtype", "scene_id"])[metrics].mean()
    mean = per_scene.groupby("dtype").mean()
    sem = per_scene.groupby("dtype").sem()
    return mean, sem


def character(cond, metrics):
    f = os.path.join(OUT, "descriptors.csv")
    d = pd.read_csv(f)
    c = cond.merge(d, on=["dataset", "scene", "method", "condition"], how="inner")
    desc = ["sharp_ratio", "err_top1", "temp_ratio"]
    low = c.sort_values("level").groupby(["scene_id", "method", "ladder"]).head(1)
    tab = low.groupby("dtype")[desc].mean()
    full = c[c.condition == "iters-30000"].groupby("method")[desc].mean()
    full.index = [m + ":full" for m in full.index]
    tab = pd.concat([tab, full])
    tab.to_csv(os.path.join(OUT, "character_by_type.csv"))
    print("\n=== character of the renderings: lowest level of every ladder, and the fully trained models ===")
    print(tab.round(3).to_string())
    z = {m: c.groupby("scene_id")[m].transform(lambda v, s=SIGN[m]: (s * v - (s * v).mean()) / (s * v).std()) for m in metrics}
    rows = []
    for a, b in itertools.combinations(metrics, 2):
        diff = z[a] - z[b]
        for k in desc:
            rs = []
            for _, g in c.groupby("scene_id"):
                v = g[k]
                if v.notna().sum() >= 5:
                    rs.append(sstats.spearmanr(diff[g.index][v.notna()], v[v.notna()]).statistic)
            if rs:
                rows.append(dict(metric_a=a, metric_b=b, descriptor=k, rho=fisher_mean(rs), n_scenes=len(rs),
                                 rho_min=float(np.min(rs)), rho_max=float(np.max(rs))))
    ax = pd.DataFrame(rows)
    ax.to_csv(os.path.join(OUT, "axis_correlation.csv"), index=False)
    print("\n=== Spearman(z_a - z_b, descriptor) inside scenes (> 0: a is more lenient than b when the descriptor is large) ===")
    print(ax[ax.metric_a.isin(["psnr"])].pivot(index="metric_b", columns="descriptor", values="rho").round(2).to_string())


def main():
    cond = load()
    print(f"{len(cond)} conditions in {cond.scene_id.nunique()} scenes")
    print(cond.groupby(["scene_id", "method", "ladder"]).size().unstack(fill_value=0).to_string())
    mono = monotonicity(cond, IMG + VID)
    mono.to_csv(os.path.join(OUT, "monotonicity.csv"), index=False)
    print("\n=== monotonicity along the ladders: mean Spearman(level, metric) / share of ladders with inversions ===")
    piv = mono.groupby(["method", "ladder", "metric"]).agg(spearman=("spearman", "mean"),
                                                            with_inversions=("inversions", lambda v: (v > 0).mean()),
                                                            n=("spearman", "size"))
    print(piv.unstack("metric")["spearman"].round(3).to_string())
    print(piv.unstack("metric")["with_inversions"].round(2).to_string())
    mean, worst = intermetric(cond, IMG, "scene_id")
    mean.to_csv(os.path.join(OUT, "intermetric_srocc.csv"))
    print("\n=== mean within-scene SROCC between metrics (conditions) ===")
    print(mean.round(3).to_string())
    frames_file = os.path.join(OUT, "frames.csv")
    fr = pd.read_csv(frames_file)
    fr = fr[fr.seed == 42]
    fr["scene_id"] = fr.dataset + "/" + fr.scene
    fr["view_id"] = fr.scene_id + "#" + fr.frame.astype(str)
    img_scene, _ = intermetric(fr, IMG, "scene_id")
    img_scene.to_csv(os.path.join(OUT, "intermetric_image.csv"))
    sub = fr[fr.frame % 10 == 0] if len(fr) > 200000 else fr
    img_view, _ = intermetric(sub, IMG, "view_id")
    img_view.to_csv(os.path.join(OUT, "intermetric_view.csv"))
    print("\n=== mean SROCC between metrics over renderings of the same view ===")
    print(img_view.round(3).to_string())
    noise, thr = seed_noise(IMG + VID)
    print(f"\n=== seed-to-seed variation ({len(noise)} conditions trained twice): median / 90th percentile of |difference| ===")
    print(pd.DataFrame({"median": noise.median(), "p90": noise.quantile(0.9), "max": noise.max()}).round(4).to_string())
    total, by_type, cases = disagreements(cond, IMG, thr)
    total.to_csv(os.path.join(OUT, "disagreement_rate.csv"), index=False)
    by_type.to_csv(os.path.join(OUT, "disagreement_by_type.csv"), index=False)
    cases[cases.resolved].head(20000).to_csv(os.path.join(OUT, "disagreement_cases.csv"), index=False)
    print("\n=== share of condition pairs ordered differently by two metrics ===")
    print(total.pivot(index="metric_a", columns="metric_b", values="rate").round(3).to_string())
    print("--- the same, counting only differences beyond the seed-to-seed variation of both metrics ---")
    print(total.pivot(index="metric_a", columns="metric_b", values="robust_rate").round(3).to_string())
    print(by_type[by_type.metric_a.isin(CORE) & by_type.metric_b.isin(CORE)]
          .pivot_table(index=["metric_a", "metric_b"], columns="kind", values="robust_rate").round(3).to_string())
    mp = matched_psnr(cond, IMG + VID)
    mp.to_csv(os.path.join(OUT, "matched_psnr.csv"), index=False)
    print(f"\n=== 3DGS vs NeRF at equal PSNR ({len(mp)} condition pairs within 0.3 dB) ===")
    cols = [c for c in mp.columns if c.startswith("d_") and mp[c].notna().any()]
    print(pd.DataFrame({"mean difference (3DGS - NeRF, >0 favours 3DGS)": mp[cols].mean(),
                        "in scene std": mp[["dz_" + c[2:] for c in cols]].mean().to_numpy(),
                        "share favouring 3DGS": (mp[cols] > 0).mean()}).round(3).to_string())
    print(by_type[by_type.metric_a.isin(CORE) & by_type.metric_b.isin(CORE)]
          .pivot_table(index=["metric_a", "metric_b"], columns="kind", values="rate").round(3).to_string())
    character(cond, IMG)
    bias, sem = type_bias(cond, CORE)
    bias.to_csv(os.path.join(OUT, "type_bias.csv"))
    sem.to_csv(os.path.join(OUT, "type_bias_sem.csv"))
    print("\n=== metric bias per degradation type (z units, + = metric more lenient than the consensus) ===")
    print(bias.round(2).to_string())
    vid = cond[cond.is_video]
    rows = []
    for m in IMG + ["cvvdp_sust"]:
        rs = [sstats.spearmanr(SIGN[m] * g[m], g.cvvdp).statistic for _, g in vid.groupby("scene_id")]
        dis = stats.pairwise_disagreement(SIGN[m] * vid[m], vid.cvvdp, vid.scene_id)
        rows.append(dict(metric=m, srocc_with_cvvdp_video=fisher_mean(rs), disagreement=dis[0], pairs=dis[1]))
    vv = pd.DataFrame(rows)
    vv.to_csv(os.path.join(OUT, "video_vs_frames.csv"), index=False)
    print("\n=== camera-path scenes: frame-averaged metrics vs ColorVideoVDP (video) ===")
    print(vv.round(3).to_string(index=False))
    tp = vid[["scene_id", "method", "condition", "ladder", "level", "cvvdp", "cvvdp_sust", "cvvdp_image"]].copy()
    tp["transient_loss"] = tp.cvvdp_sust - tp.cvvdp
    tp.to_csv(os.path.join(OUT, "transient_loss.csv"), index=False)
    print("\n=== JOD removed by the transient channel (video mode without it minus video mode), mean over scenes ===")
    print(tp.pivot_table(index=["method", "ladder"], columns="condition", values="transient_loss").round(2)
          .to_string(na_rep=""))
if __name__ == "__main__":
    main()
