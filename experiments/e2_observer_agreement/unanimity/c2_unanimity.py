#!/usr/bin/env python
import itertools
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
OUT = P.UNANIMITY
RNG = np.random.default_rng(20261124)
NBOOT = 2000
SIGN = {"psnr": 1, "ssim": 1, "ms_ssim": 1, "lpips_alex": -1, "lpips_vgg": -1, "cvvdp_image": 1, "flip": -1, "cvvdp": 1,
        "stlpips_alex": -1, "pub_DISTS": -1, "pub_FSIM": 1, "pub_VIF": 1, "pub_VMAF": 1, "pub_FVVDP": 1}
LABEL = {"psnr": "PSNR", "ssim": "SSIM", "ms_ssim": "MS-SSIM", "lpips_alex": "LPIPS-Alex", "lpips_vgg": "LPIPS-VGG",
         "cvvdp_image": "CVVDP (image)", "flip": "FLIP", "cvvdp": "CVVDP (video)", "stlpips_alex": "ST-LPIPS-Alex",
         "pub_DISTS": "DISTS [pub]", "pub_FSIM": "FSIM [pub]", "pub_VIF": "VIF [pub]", "pub_VMAF": "VMAF [pub]",
         "pub_FVVDP": "FVVDP [pub]"}
VOTES = {"triplet_vgg": ["psnr", "ssim", "lpips_vgg"],
         "triplet_alex": ["psnr", "ssim", "lpips_alex"],
         "five_family": ["psnr", "ssim", "lpips_alex", "flip", "cvvdp"]}
EXIST = {"psnr": "PSNR", "ssim": "SSIM", "ms_ssim": "MS-SSIM", "lpips_alex": "LPIPS (Alex)", "lpips_vgg": "LPIPS (VGG)",
         "flip": "FLIP", "cvvdp_image": "ColorVideoVDP (image)", "cvvdp": "ColorVideoVDP (video)"}


def load():
    g = pd.read_csv(os.path.join(P.GSQA_RESULTS, "metrics.csv"))
    n = pd.read_csv(os.path.join(P.NVSQA_RESULTS, "metrics.csv"))
    g["dataset"], n["dataset"] = "GS-QA", "NVS-QA"
    return {"GS-QA": g, "NVS-QA": n}


def pairs_of(df, metrics):
    rows = []
    for scene, d in df.groupby("scene", sort=True):
        d = d.reset_index(drop=True)
        for i, j in itertools.combinations(range(len(d)), 2):
            r = dict(scene=scene, scene_class=d.scene_class[i], a=d.method[i], b=d.method[j],
                     ddmos=float(d.dmos[i] - d.dmos[j]))
            for m in metrics:
                r[m] = SIGN[m] * float(d[m][i] - d[m][j])
                r["raw_" + m] = float(d[m][i] - d[m][j])
            rows.append(r)
    p = pd.DataFrame(rows)
    return p[p.ddmos != 0].reset_index(drop=True)


def acc(vote, human):
    vote, human = np.asarray(vote, float), np.asarray(human, float)
    if len(vote) == 0:
        return np.nan
    s = np.sign(vote) * np.sign(human)
    return float(np.mean(np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))))


def zscore_within(df, metrics):
    z = pd.DataFrame(index=df.index)
    for m in metrics:
        v = SIGN[m] * df[m]
        gb = v.groupby(df.scene)
        sd = gb.transform("std").replace(0, np.nan)
        z["z_" + m] = ((v - gb.transform("mean")) / sd).fillna(0.0)
        z["r_" + m] = gb.rank()
    return z


def add_consensus(df):
    df = df.copy()
    allm = sorted({m for v in VOTES.values() for m in v})
    z = zscore_within(df, allm)
    for name, ms in VOTES.items():
        df["cons_z_" + name] = z[["z_" + m for m in ms]].mean(axis=1)
        df["cons_r_" + name] = z[["r_" + m for m in ms]].mean(axis=1)
        SIGN["cons_z_" + name] = SIGN["cons_r_" + name] = 1
        LABEL["cons_z_" + name] = f"consensus z-mean ({name})"
        LABEL["cons_r_" + name] = f"consensus rank-mean ({name})"
    return df


def within_srocc(df, m):
    out = []
    for _, d in df.groupby("scene"):
        if d[m].nunique() > 1 and d.dmos.nunique() > 1:
            out.append(spearmanr(SIGN[m] * d[m], d.dmos).statistic)
    return float(np.mean(out)) if out else np.nan


def vote_stats(p, ms):
    v = np.sign(p[ms].to_numpy())
    npos, nneg = (v > 0).sum(1), (v < 0).sum(1)
    maj = np.sign(npos - nneg)
    level = np.maximum(npos, nneg)
    return maj, level


def summarise_vote(p, name, ms):
    maj, level = vote_stats(p, ms)
    k = len(ms)
    un = level == k
    return dict(vote=name, n_pairs=len(p), share_unanimous=float(un.mean()),
                acc_unanimous=acc(maj[un], p.ddmos[un]), n_unanimous=int(un.sum()),
                acc_split_majority=acc(maj[~un], p.ddmos[~un]), n_split=int((~un).sum()),
                acc_majority_all=acc(maj, p.ddmos),
                **{f"acc_split_{m}": acc(p[m][~un], p.ddmos[~un]) for m in ms})


def correct(vote, human):
    s_ = np.sign(np.asarray(vote, float)) * np.sign(np.asarray(human, float))
    return np.where(s_ > 0, 1.0, np.where(s_ < 0, 0.0, 0.5))


def boot_weights(scene_codes, n_scenes):
    w = RNG.multinomial(n_scenes, np.full(n_scenes, 1.0 / n_scenes), size=NBOOT).astype(float)
    return w[:, scene_codes]


def wmean(w, x, mask=None):
    m = np.ones(len(x)) if mask is None else np.asarray(mask, float)
    den = w @ m
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (w @ (m * x)) / den, np.nan)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)), float(np.mean(x > 0))) if len(x) else (np.nan,) * 3


def main():
    data = load()
    res = {}
    single_rows, vote_rows, level_rows, cons_rows, boot_rows, tie_rows, check_rows, strat_rows = [], [], [], [], [], [], [], []
    sn = pd.read_csv(os.path.join(P.PART1_RESULTS, "seed_noise.csv"))
    thr = {m: float(sn[m].quantile(0.9)) for m in ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg", "flip", "cvvdp_image", "cvvdp"]}
    res["seed_noise_p90"] = thr
    for ds, df in data.items():
        df = add_consensus(df)
        base = [m for m in SIGN if m in df.columns and not m.startswith("cons_")]
        cons = [m for m in df.columns if m.startswith("cons_")]
        metrics = base + cons
        p_all = pairs_of(df, metrics)
        ex = pd.read_csv(os.path.join(P.subjective_results('gsqa' if ds == 'GS-QA' else 'nvsqa'), "agreement_all.csv")).set_index("metric")
        for m, name in EXIST.items():
            big = p_all[p_all.ddmos.abs() > 0.5]
            check_rows.append(dict(dataset=ds, metric=m, acc_all_here=acc(p_all[m], p_all.ddmos), acc_all_existing=ex.loc[name, "pair_acc_d0"],
                                   n_all_here=len(p_all), n_all_existing=ex.loc[name, "pairs_d0"],
                                   acc_d05_here=acc(big[m], big.ddmos), acc_d05_existing=ex.loc[name, "pair_acc_d0.5"],
                                   n_d05_here=len(big), n_d05_existing=ex.loc[name, "pairs_d0.5"],
                                   within_srocc_here=within_srocc(df, m), within_srocc_existing=ex.loc[name, "srocc_within_scene"]))
        subsets = {"all": np.ones(len(p_all), bool)}
        for c in sorted(p_all.scene_class.unique()):
            subsets[c] = (p_all.scene_class == c).to_numpy()
        if ds == "NVS-QA":
            subsets["real"] = p_all.scene_class.isin(["FF", "360"]).to_numpy()
            subsets["synthetic"] = p_all.scene_class.isin(["FF-synthetic", "360-synthetic"]).to_numpy()
        for sub, mask in subsets.items():
            for dname, dmin in (("all pairs", 0.0), ("|dDMOS|>0.5", 0.5)):
                p = p_all[mask & (p_all.ddmos.abs() > dmin).to_numpy()].reset_index(drop=True)
                if len(p) == 0:
                    continue
                dsub = df[df.scene.isin(p.scene.unique())]
                for m in metrics:
                    row = dict(dataset=ds, subset=sub, pairs=dname, metric=m, label=LABEL[m], n_pairs=len(p), n_scenes=p.scene.nunique(),
                               pair_acc=acc(p[m], p.ddmos), within_srocc=within_srocc(dsub, m) if dmin == 0 else np.nan)
                    (cons_rows if m.startswith("cons_") else single_rows).append(row)
                for name, ms in VOTES.items():
                    vote_rows.append(dict(dataset=ds, subset=sub, pairs=dname, n_scenes=p.scene.nunique(), **summarise_vote(p, name, ms)))
                    maj, level = vote_stats(p, ms)
                    for lv in sorted(set(level)):
                        k = level == lv
                        level_rows.append(dict(dataset=ds, subset=sub, pairs=dname, vote=name, majority_size=f"{lv}-{len(ms) - lv}",
                                               n=int(k.sum()), share=float(k.mean()), acc_majority=acc(maj[k], p.ddmos[k])))
                if sub in ("all", "real", "synthetic", "FF", "360", "FF-synthetic", "360-synthetic"):
                    single = {m: acc(p[m], p.ddmos) for m in base if m in VOTES["five_family"] + ["lpips_vgg", "cvvdp_image", "ms_ssim"]}
                    best = max(single, key=single.get)
                    codes = pd.factorize(p.scene)[0]
                    w = boot_weights(codes, codes.max() + 1)
                    hum = p.ddmos.to_numpy()
                    for name, ms in VOTES.items():
                        maj, level = vote_stats(p, ms)
                        un = (level == len(ms)).astype(float)
                        cm = correct(maj, hum)
                        q_point, q_boot = {}, {}
                        def put(k, point, boot):
                            q_point[k], q_boot[k] = point, boot
                        ones = np.ones(len(p))
                        put("a_un", acc(maj[un > 0], hum[un > 0]), wmean(w, cm, un))
                        put("a_sp", acc(maj[un == 0], hum[un == 0]), wmean(w, cm, 1 - un))
                        put("gap", q_point["a_un"] - q_point["a_sp"], q_boot["a_un"] - q_boot["a_sp"])
                        put("share", float(un.mean()), wmean(w, un))
                        lp = [m for m in ms if m.startswith("lpips")][0]
                        c_lp, c_ps, c_ss = correct(p[lp], hum), correct(p["psnr"], hum), correct(p["ssim"], hum)
                        sp = 1 - un
                        if sp.sum() > 0:
                            put("split_acc_lpips", float((c_lp * sp).sum() / sp.sum()), wmean(w, c_lp, sp))
                            put("split_acc_psnr", float((c_ps * sp).sum() / sp.sum()), wmean(w, c_ps, sp))
                            put("split_acc_ssim", float((c_ss * sp).sum() / sp.sum()), wmean(w, c_ss, sp))
                            put("split_lpips_minus_psnr", q_point["split_acc_lpips"] - q_point["split_acc_psnr"], q_boot["split_acc_lpips"] - q_boot["split_acc_psnr"])
                            put("split_lpips_minus_ssim", q_point["split_acc_lpips"] - q_point["split_acc_ssim"], q_boot["split_acc_lpips"] - q_boot["split_acc_ssim"])
                        for c in ("cons_z_" + name, "cons_r_" + name):
                            cc = correct(p[c], hum)
                            for other, tag in ((correct(p[best], hum), "best"), (correct(p["lpips_alex"], hum), "lpips_alex"),
                                               (correct(p["psnr"], hum), "psnr"), (cm, "majority")):
                                put(f"{c}_minus_{tag}", float(cc.mean() - other.mean()), wmean(w, cc - other))
                        for k in q_point:
                            lo, hi, ppos = ci(q_boot[k])
                            boot_rows.append(dict(dataset=ds, subset=sub, pairs=dname, vote=name, quantity=k, best_single=best,
                                                  value=q_point[k], lo=lo, hi=hi, share_boot_positive=ppos, n_pairs=len(p), n_scenes=p.scene.nunique()))
            if sub in ("all", "real", "synthetic"):
                edges = [0, 0.25, 0.5, 1.0, np.inf]
                for name, ms in VOTES.items():
                    maj, level = vote_stats(p_all[mask], ms)
                    q = p_all[mask].reset_index(drop=True)
                    un = level == len(ms)
                    for lo_, hi_ in zip(edges[:-1], edges[1:]):
                        k = ((q.ddmos.abs() > lo_) & (q.ddmos.abs() <= hi_)).to_numpy()
                        strat_rows.append(dict(dataset=ds, subset=sub, vote=name, ddmos_bin=f"({lo_}, {hi_}]", n=int(k.sum()),
                                               share_unanimous=float(un[k].mean()) if k.sum() else np.nan,
                                               n_unanimous=int((un & k).sum()), acc_unanimous=acc(maj[un & k], q.ddmos[un & k]),
                                               n_split=int((~un & k).sum()), acc_split_majority=acc(maj[~un & k], q.ddmos[~un & k])))
            p = p_all[mask].reset_index(drop=True)
            for dname, dmin in (("all pairs", 0.0), ("|dDMOS|>0.5", 0.5)):
                q = p[p.ddmos.abs() > dmin]
                for m in thr:
                    dec = q["raw_" + m].abs() >= thr[m]
                    tie_rows.append(dict(dataset=ds, subset=sub, pairs=dname, metric=m, threshold=thr[m], n_pairs=len(q), coverage=float(dec.mean()),
                                         acc_decided=acc(q[m][dec], q.ddmos[dec]), acc_undecided=acc(q[m][~dec], q.ddmos[~dec]), acc_all=acc(q[m], q.ddmos)))
                for name, ms in VOTES.items():
                    v = np.sign(q[ms].to_numpy()) * np.stack([(q["raw_" + m].abs() >= thr[m]).to_numpy() for m in ms], 1)
                    npos, nneg = (v > 0).sum(1), (v < 0).sum(1)
                    un = ((npos == 0) | (nneg == 0)) & ((npos + nneg) >= 2)
                    conflict = (npos > 0) & (nneg > 0)
                    none = (npos + nneg) < 2
                    maj = np.sign(npos - nneg)
                    tie_rows.append(dict(dataset=ds, subset=sub, pairs=dname, metric="vote:" + name, threshold=np.nan, n_pairs=len(q),
                                         coverage=float(un.mean()), acc_decided=acc(maj[un], q.ddmos[un]),
                                         acc_undecided=acc(maj[conflict], q.ddmos[conflict]), acc_all=acc(maj, q.ddmos),
                                         share_conflict=float(conflict.mean()), share_fewer_than_two_decide=float(none.mean())))
    tabs = dict(single_metric_accuracy=single_rows, vote_accuracy=vote_rows, vote_levels=level_rows, consensus=cons_rows,
                bootstrap_contrasts=boot_rows, tie_rule=tie_rows, check_against_existing=check_rows, stratified_by_ddmos=strat_rows)
    T = {}
    for k, rows in tabs.items():
        T[k] = pd.DataFrame(rows)
        T[k].to_csv(os.path.join(OUT, k + ".csv"), index=False)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400, "display.float_format", lambda x: f"{x:.3f}")
    print("== implementation check (here vs existing agreement_all.csv)")
    print(T["check_against_existing"].to_string(index=False))
    for k in ("single_metric_accuracy", "consensus"):
        print("\n==", k)
        print(T[k][T[k].subset.isin(["all", "real", "synthetic"])].pivot_table(index=["dataset", "subset", "label"], columns="pairs",
                                                                             values=["pair_acc", "within_srocc", "n_pairs"], sort=False).to_string())
    print("\n== votes")
    print(T["vote_accuracy"][["dataset", "subset", "pairs", "vote", "n_pairs", "n_scenes", "share_unanimous", "n_unanimous", "acc_unanimous",
                              "n_split", "acc_split_majority", "acc_majority_all"]].to_string(index=False))
    print("\n== vote levels (all pairs)")
    print(T["vote_levels"][(T["vote_levels"].pairs == "all pairs") & T["vote_levels"].subset.isin(["all", "real", "synthetic"])].to_string(index=False))
    print("\n== bootstrap over scenes")
    b = T["bootstrap_contrasts"]
    print(b[b.quantity.isin(["gap", "a_un", "a_sp"]) & b.subset.isin(["all", "real", "synthetic", "FF", "360"])].to_string(index=False))
    print(b[b.quantity.str.contains("minus") & b.subset.isin(["all"])].to_string(index=False))
    print("\n== seed-noise tie rule")
    print(T["tie_rule"][T["tie_rule"].subset.isin(["all"])].to_string(index=False))
    return T, res
if __name__ == "__main__":
    main()
