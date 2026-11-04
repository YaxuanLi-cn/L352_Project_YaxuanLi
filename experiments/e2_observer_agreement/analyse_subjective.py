#!/usr/bin/env python
import argparse
import json
import os
import sys
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import numpy as np              
import pandas as pd              
from nvsqa import stats              
from nvsqa.metrics import LABEL, SIGN              
OURS = ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg", "flip", "cvvdp_image", "cvvdp", "cvvdp_sust"]
PUB_SIGN = {"FSIM": 1, "FVVDP": 1, "GMSD": -1, "IW-SSIM": 1, "MAD": -1, "MSE-RGB": -1, "MS-SSIM": 1, "NLPD": -1,
            "PSNR-HVS": 1, "PSNR-Y": 1, "PSNR-YUV": 1, "SSIM": 1, "VIF": 1, "VIFp": 1, "VSI": 1, "DISTS": -1,
            "LPIPS": -1, "ST-LPIPS": -1, "ST-LPIPS (sheet order)": -1, "VMAF": 1}
EXTRA = ["stlpips_alex", "stlpips_vgg"]
TAG = {"gsqa": "GS-QA", "gsqa_cc": "GS-QA", "nvsqa": "NVS-QA"}


def oriented(df, include_pub=True, tag="GS-QA"):
    preds = {LABEL[m]: SIGN[m] * df[m].to_numpy(float) for m in OURS + EXTRA if m in df and df[m].notna().all()}
    if include_pub:
        for m, s in PUB_SIGN.items():
            col = "pub_" + m
            if col in df and df[col].notna().all():
                preds[f"{m} [{tag}]"] = s * df[col].to_numpy(float)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gsqa")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    d = P.subjective_results(args.dataset)
    df = pd.read_csv(os.path.join(d, "metrics.csv"))
    summary = {"n": len(df), "subsets": {}}
    tag = TAG.get(args.dataset, args.dataset)
    subsets = {"all": df}
    for cls in sorted(df.scene_class.unique()):
        subsets[cls] = df[df.scene_class == cls]
    if df.scene_class.str.contains("synthetic").any():
        subsets["real"] = df[~df.scene_class.str.contains("synthetic")]
        subsets["synthetic"] = df[df.scene_class.str.contains("synthetic")]
    for name, sub in subsets.items():
        sub = sub.reset_index(drop=True)
        preds = oriented(sub, tag=tag)
        y = sub.dmos.to_numpy(float)
        summ, samples, names = stats.bootstrap_agreement(preds, y, n_boot=args.n_boot, seed=0)
        csumm, csamples, _ = stats.bootstrap_agreement(preds, y, n_boot=args.n_boot, seed=1,
                                                       groups=sub.scene.to_numpy(), measures=("srocc",))
        rows = []
        for k in names:
            r = {"metric": k, "n": len(sub)}
            for m, (est, lo, hi) in summ[k].items():
                r[m], r[m + "_lo"], r[m + "_hi"] = est, lo, hi
            r["srocc_scene_lo"], r["srocc_scene_hi"] = csumm[k]["srocc"][1], csumm[k]["srocc"][2]
            wg = stats.within_group_test(preds[k], y, sub.scene, n_perm=5000)
            r["srocc_within_scene"], r["within_p"] = wg["mean"], wg["p"]
            r["within_lo"], r["within_hi"] = wg["lo"], wg["hi"]
            for thr in (0.0, 0.5):
                acc, npairs = stats.pairwise_accuracy(preds[k], y, sub.scene, min_delta=thr)
                r[f"pair_acc_d{thr:g}"], r[f"pairs_d{thr:g}"] = acc, npairs
            rows.append(r)
        tab = pd.DataFrame(rows).sort_values("srocc", ascending=False)
        tab.to_csv(os.path.join(d, f"agreement_{name}.csv"), index=False)
        diffs = stats.paired_difference(samples, names, "srocc")
        cdiffs = stats.paired_difference(csamples, names, "srocc")
        pd.DataFrame([dict(a=a, b=b, **v, scene_lo=cdiffs[(a, b)]["lo"], scene_hi=cdiffs[(a, b)]["hi"],
                           scene_p=cdiffs[(a, b)]["p"]) for (a, b), v in diffs.items()]).to_csv(
            os.path.join(d, f"differences_{name}.csv"), index=False)
        summary["subsets"][name] = tab.to_dict(orient="records")
        print(f"\n=== {args.dataset} / {name} (n={len(sub)}) ===")
        print(tab[["metric", "srocc", "srocc_lo", "srocc_hi", "krocc", "plcc", "rmse", "srocc_within_scene",
                   "within_p", "pair_acc_d0", "pair_acc_d0.5"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    within_var = df.groupby("scene").dmos.transform(lambda v: v - v.mean()).var()
    summary["dmos_variance"] = dict(total=float(df.dmos.var()), within_scene_share=float(within_var / df.dmos.var()))
    print(f"\nDMOS variance: {100 * within_var / df.dmos.var():.0f} % within scenes, "
          f"{100 * (1 - within_var / df.dmos.var()):.0f} % between scenes")
    preds = oriented(df, tag=tag)
    y = df.dmos.to_numpy(float)
    ws = pd.DataFrame({k: stats.within_group_srocc(v, y, df.scene) for k, v in preds.items()}).T
    ws["mean"] = ws.mean(axis=1)
    ws.to_csv(os.path.join(d, "within_scene.csv"))
    print("\n=== SROCC within each scene (8 methods per scene) ===")
    print(ws.to_string(float_format=lambda v: f"{v:.2f}"))
    wm = pd.DataFrame({k: stats.within_group_srocc(v, y, df.method) for k, v in preds.items()}).T
    wm["mean"] = wm.mean(axis=1)
    wm.to_csv(os.path.join(d, "within_method.csv"))
    print("\n=== SROCC within each rendering method (across scenes) ===")
    print(wm.to_string(float_format=lambda v: f"{v:.2f}"))
    summary["within_method"] = json.loads(wm.to_json())
    res = {}
    for k, v in preds.items():
        f = stats.fit_logistic(v, y)
        res[k] = pd.Series(y - f(v)).groupby(df.method.to_numpy()).mean()
    res = pd.DataFrame(res).T
    res.to_csv(os.path.join(d, "per_method_residuals.csv"))
    print("\n=== mean residual DMOS - predicted, per method ===")
    print(res.to_string(float_format=lambda v: f"{v:+.2f}"))
    summary["within_scene"] = json.loads(ws.to_json())
    summary["per_method_residuals"] = json.loads(res.to_json())
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
if __name__ == "__main__":
    main()
