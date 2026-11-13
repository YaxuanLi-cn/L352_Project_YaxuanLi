#!/usr/bin/env python
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
REC = P.rec(P.E5, P.S1)
IMG = ["l1", "mse", "psnr", "ssim", "ms_ssim", "lpips_vgg", "lpips_alex", "dists", "flip", "cvvdp_image", "dreamsim", "fsim", "vif"]
SIGN = {"l1": -1, "mse": -1, "psnr": 1, "ssim": 1, "ms_ssim": 1, "lpips_vgg": -1, "lpips_alex": -1, "dists": -1, "flip": -1,
        "cvvdp_image": 1, "dreamsim": -1, "fsim": 1, "vif": 1}
SM =pd.read_csv(os.path.join(P.ORIG_TABLES, "column_scale.csv")).set_index("metric").scale
SEED_P90 = dict(psnr=0.45, ssim=0.008, lpips_vgg=0.011)


def main():
    sdn = sys.argv[1] if len(sys.argv) > 1 else "gsqa_truck"
    spec = sdn.replace("_", "/", 1)
    ref = pd.read_csv(os.path.join(P.RERUN_RESULTS, "metrics.csv"))
    ref = ref[(ref.scene == spec) & (ref.loss == "default") & (ref.variant == "ctrl") & (ref.seed == 0) & (ref.split == "test")]
    ref = ref.pivot_table(index="view_id", columns="metric", values="value")
    rows = []
    for d in sorted(glob.glob(os.path.join(P.S1_RUNS, sdn, "*"))):
        name = os.path.basename(d)
        if not (name.startswith("mad") or name.startswith("front-") or name.startswith("ctrl__")) or not os.path.exists(os.path.join(d, "done.json")) or "smoke" in name:
            continue
        info = json.load(open(os.path.join(d, "done.json")))
        m = pd.read_parquet(os.path.join(d, "metrics.parquet"))
        t = m[m.split == "test"].pivot_table(index="view", columns="metric", values="value")
        base = ref.loc[t.index]
        row = dict(run=name, status=info["status"], n_views=len(t))
        for k in IMG:
            dlt = (t[k] - base[k]).mean()
            row[f"d_{k}"] = dlt
            row[f"z_{k}"] = SIGN[k] * dlt / SM[k]
        hold = info.get("loss", "").startswith("mad:") and (name.split("__hold-")[1].split("__")[0].split("-") if "__hold-" in name else ["psnr", "ssim", "lpips_vgg"]) or []
        hold =[h if h != "lpips" else "lpips_vgg" for h in hold]
        row["held"] = ",".join(hold)
        for k in ("psnr", "ssim", "lpips_vgg"):
            row[f"{k}_in_seedP90"] = abs(row[f"d_{k}"]) / SEED_P90[k]
            row[f"{k}_perview_absdev"] = float((t[k] - base[k]).abs().mean())
        free = [k for k in IMG if k not in ("l1", "mse", "psnr", "ssim", "ms_ssim", "lpips_vgg", "lpips_alex")]
        row["z_free_mean"] = float(np.mean([row[f"z_{k}"] for k in free]))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(REC, f"mad_summary_{sdn}.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    show = ["run", "d_psnr", "d_ssim", "d_lpips_vgg", "psnr_in_seedP90", "ssim_in_seedP90", "lpips_vgg_in_seedP90", "d_cvvdp_image",
            "d_dists", "z_dists", "z_flip", "z_cvvdp_image", "z_dreamsim", "z_fsim", "z_vif", "z_lpips_alex", "z_ms_ssim", "z_free_mean"]
    print(df[show].round(3).T.to_string())
    print("written", out)
if __name__ == "__main__":
    main()
