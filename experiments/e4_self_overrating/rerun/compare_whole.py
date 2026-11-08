#!/usr/bin/env python
import argparse
import itertools
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import helpers as C              
import numpy as np              
import pandas as pd              
from scipy.stats import kendalltau              
GH = os.path.dirname(P.ORIG_RESULTS)
COMMON7 = ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg", "flip", "cvvdp_image"]
TABLES = [
    ("results", "self_overestimation_summary.csv", ["split", "variant"], "mean", "lo", "hi", "sd_seed", None,
     lambda d: d[d.variant.isin(["pure", "mixed"])]),
    ("results", "self_overestimation.csv", ["split", "cond"], "score", "lo", "hi", "sd_seed", "finding",
     lambda d: d[(d.reference == "metrics") & (d.kind == "self")]),
    ("results", "self_overestimation.csv", ["split", "reference", "cond", "metric"], "score", "lo", "hi", "sd_seed",
     "finding", lambda d: d[~((d.reference == "metrics") & (d.kind == "self"))]),
    ("results", "cross_matrix.csv", ["split", "cond", "metric"], "z", "z_lo", "z_hi", "sd_seed_z", "finding", None),
    ("results", "overrating_matrix.csv", ["split", "cond", "metric"], "score", "lo", "hi", "sd_seed", "finding", None),
    ("results", "generalisation_gap.csv", ["cond", "metric"], "gap", "lo", "hi", "sd_seed", "finding", None),
    ("results", "kendall_tau_diff.csv", ["m1", "m2"], "finetune", None, None, None, None, None),
    ("results", "kendall_tau_diff.csv", ["m1", "m2"], "diff", "lo", "hi", None, None, None),
    ("results", "concordance_by_cause.csv", ["pair_class", "range_matched", "m1", "m2"], "concordance", "lo", "hi",
     None, None, None),
]
LABELS = {
    ("self_overestimation_summary.csv", "mean"): "S of the two loss groups",
    ("self_overestimation.csv", "score", 1): "S per loss model",
    ("self_overestimation.csv", "score", 2): "same-family transfer and other reference sets",
    ("cross_matrix.csv", "z"): "cross matrix z",
    ("overrating_matrix.csv", "score"): "overrating matrix",
    ("generalisation_gap.csv", "gap"): "generalisation gap (training minus held-out views)",
    ("kendall_tau_diff.csv", "finetune"): "Kendall tau of the loss models",
    ("kendall_tau_diff.csv", "diff"): "Kendall tau, loss models minus natural degradations",
    ("concordance_by_cause.csv", "concordance"): "concordance by cause",
}


def load(root, sub, name, flt):
    d = pd.read_csv(os.path.join(root, sub, "tables", name))
    return flt(d) if flt else d


def compare(old, new, keys, val, lo, hi, sd, fnd, n_scenes):
    o, n = old.copy(), new.copy()
    for k in keys:
        o[k] = o[k].fillna("").astype(str)
        n[k] = n[k].fillna("").astype(str)
    cols = [c for c in (val, lo, hi, sd, fnd) if c]
    m = o[keys + cols].merge(n[keys + cols], on=keys, how="outer", suffixes=("_paper", "_new"), indicator=True)
    m["diff"] = m[val + "_new"] - m[val + "_paper"]
    if lo:
        has = m[lo + "_paper"].notna() & m[hi + "_paper"].notna()
        inside = (m[val + "_new"] >= m[lo + "_paper"]) & (m[val + "_new"] <= m[hi + "_paper"])
        m["new_in_paper_ci"] = np.where(has, inside.astype(float), np.nan)
    if sd:
        m["diff_over_sd_seed"] = m["diff"].abs() / m[sd + "_paper"]
        m["within_2sd_seed"] = m["diff"].abs() <= 2 * m[sd + "_paper"]
        m["within_2sd_seed_mean"] = m["diff"].abs() <= 2 * m[sd + "_paper"] / np.sqrt(n_scenes)
    if fnd:
        m["finding_changed"] = m[fnd + "_paper"].astype(str) != m[fnd + "_new"].astype(str)
    return m


def truck_tau_seed_sd(metrics_csv):
    df = pd.read_csv(metrics_csv)
    df = df[(df.scene == "gsqa/truck") & (df.split == "test") & df.metric.isin(COMMON7) & (df.variant != "base")]
    sign = {"psnr": 1, "ssim": 1, "ms_ssim": 1, "lpips_alex": -1, "lpips_vgg": -1, "flip": -1, "cvvdp_image": 1}
    out = {}
    for s, g in df.groupby("seed"):
        t = g.groupby(["loss", "variant", "metric"]).value.mean().unstack("metric")
        out[int(s)] = float(np.mean([kendalltau(sign[a] * t[a], sign[b] * t[b]).statistic
                                     for a, b in itertools.combinations(COMMON7, 2)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default=P.RERUN_WHOLE)
    ap.add_argument("--out", default=P.RERUN_COMPARE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n_scenes = 8
    summary, flips = [], []
    seen = {}
    for sub, name, keys, val, lo, hi, sd, fnd, flt in TABLES:
        old = load(GH, sub, name, flt)
        new = load(args.new, sub, name, flt)
        m = compare(old, new, keys, val, lo, hi, sd, fnd, n_scenes)
        seen[name] = seen.get(name, 0) + 1
        tag = f"{name[:-4]}__{val}" + (f"_{seen[name]}" if name == "self_overestimation.csv" else "")
        m.to_csv(os.path.join(args.out, f"{tag}.csv"), index=False)
        label = LABELS.get((name, val, seen[name]), LABELS.get((name, val), tag))
        both = m[m._merge == "both"]
        row = dict(table=label, file=f"{tag}.csv", cells=len(both), missing=int((m._merge != "both").sum()),
                   median_abs_diff=float(both["diff"].abs().median()), max_abs_diff=float(both["diff"].abs().max()))
        if lo:
            row["share_new_in_paper_ci"] = float(both.new_in_paper_ci.mean())
        if sd:
            row["share_within_2sd_seed"] = float(both.within_2sd_seed.mean())
            row["share_within_2sd_seed_over_sqrt8"] = float(both.within_2sd_seed_mean.mean())
            row["median_abs_diff_over_sd_seed"] = float(both.diff_over_sd_seed.median())
        if fnd:
            ch = both[both.finding_changed]
            row["findings_changed"] = len(ch)
            for _, r in ch.iterrows():
                flips.append(dict(table=label, **{k: r[k] for k in keys}, paper=r[val + "_paper"], new=r[val + "_new"],
                                  paper_ci=f"[{r[lo + '_paper']:.3f}, {r[hi + '_paper']:.3f}]",
                                  new_ci=f"[{r[lo + '_new']:.3f}, {r[hi + '_new']:.3f}]",
                                  sd_seed_paper=r[sd + "_paper"], finding_paper=r[fnd + "_paper"], finding_new=r[fnd + "_new"]))
        summary.append(row)
    pd.DataFrame(summary).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.DataFrame(flips).to_csv(os.path.join(args.out, "findings_changed.csv"), index=False)
    rows = []
    for name, f in (("paper", os.path.join(GH, "results", "metrics.csv")),
                    ("new", os.path.join(args.new, "results", "metrics.csv"))):
        t = truck_tau_seed_sd(f)
        rows.append(dict(models=name, **{f"tau_seed{k}": v for k, v in t.items()},
                         sd_seed=float(np.std(list(t.values()), ddof=1))))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "kendall_tau_truck_seeds.csv"), index=False)
    C.log(f"comparison written to {args.out}: {len(summary)} tables, {len(flips)} findings changed")
if __name__ == "__main__":
    main()
