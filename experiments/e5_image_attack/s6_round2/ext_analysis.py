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
sys.path.insert(0, P.code(P.E5, P.S4))                                
C8 = P.code(P.E5, P.S2)                                                 
W = os.path.join(P.S5_REC, "witness")
SIGN = {"psnr": 1, "ssim": 1, "ms_ssim": 1, "fsim": 1, "vif": 1, "cvvdp_image": 1, "l1": -1, "mse": -1, "lpips_vgg": -1, "lpips_alex": -1,
        "dists": -1, "flip": -1, "dreamsim": -1}
HELD = ["psnr", "ssim", "lpips_vgg", "dreamsim"]
OUTSIDE = ["flip", "cvvdp_image", "lpips_alex", "dists", "ms_ssim", "fsim", "vif"]


def main():
    import colour as CL                  
    thr = pd.read_csv(P.THRESHOLDS).set_index("metric").p90
    fl = pd.read_csv(os.path.join(P.S4_REC, "results", "floors.csv"))
    floor = fl[fl.kind.str.startswith("existing")].set_index("scene").rms_chroma
    runs = sorted(glob.glob(os.path.join(P.S6_RUNS, "*", "*", "done.json")))
    if "--compute" in sys.argv:
        sys.path.insert(0, C8)
        import aux_existing as A
        metrics = A.R.X.Metrics()
        for f in runs:
            d = os.path.dirname(f)
            if os.path.exists(os.path.join(d, "metrics.parquet")):
                continue
            j = json.load(open(f))
            sdn = j["scene"].replace("/", "_")
            held = np.asarray(j["heldout"])
            ref = A.R.C.read_pngs([os.path.join(A.RERUN, sdn, "reference", "test", f"{int(v):03d}.png") for v in held])
            A.table(metrics, j["scene"], os.path.basename(d), os.path.join(d, "test"), held, ref).to_parquet(os.path.join(d, "metrics.parquet"))
            print("computed", d)
    rows = []
    for f in runs:
        d = os.path.dirname(f)
        j = json.load(open(f))
        s = j["scene"]
        sdn = s.replace("/", "_")
        extra = list(j.get("hold_extra", []))
        row = dict(scene=s, target=j["target"], gamma=j["gamma"], wdefault=float(j.get("wdefault", 1.0)), hold_extra="+".join(extra),
                   run=os.path.basename(d), status=j.get("status"))
        mp = os.path.join(d, "metrics.parquet")
        if os.path.exists(mp):
            t = pd.read_parquet(mp)
            a = t[t.res == "full"].groupby("metric").value.mean()
            c = pd.read_parquet(os.path.join(W, f"{sdn}__control.parquet"))
            c = c[c.res == "full"].groupby("metric").value.mean()
            for m in SIGN:
                row[m] = float(-SIGN[m] * (a[m] - c[m]) / thr[m])                              
            row["held"] = all(row[m] < 1 for m in HELD + extra)
            row["held4"] = all(row[m] < 1 for m in HELD)
            out = [m for m in OUTSIDE if m not in extra]
            row["n_outside_beyond"] = int(sum(row[m] > 1 for m in out))
            row["outside_beyond"] = ",".join(m for m in out if row[m] > 1)
        elif os.path.exists(os.path.join(d, "quick.json")):                        
            q = json.load(open(os.path.join(d, "quick.json")))
            c = pd.read_parquet(os.path.join(W, f"{sdn}__control.parquet"))
            c = c[c.res == "full"].groupby("metric").value.mean()
            for m in HELD:
                row[m] = float(-SIGN[m] * (q[m] - c[m]) / thr[m])
            row["held"] = all(row[m] < 1 for m in HELD)
            row["provisional"] = True
        st = CL.stats(os.path.join(d, "test"), CL.ctrl_dir(s), [int(v) for v in j["heldout"]])
        row["tint_rms"] = st["rms_chroma"]
        row["tint"] = st["rms_chroma"] / float(floor[s])
        row["area_chroma_gt6"] = st["area_chroma_gt6"]
        if "held" in row:
            row["adversarial"] = bool(row["held"] and row["tint"] >= 2.0)
        rows.append(row)
    os.makedirs(os.path.join(P.S6_REC, "results"), exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["scene", "hold_extra", "wdefault", "target", "gamma"])
    df.to_csv(os.path.join(P.S6_REC, "results", "ext_runs.csv"), index=False)
    L = ["| scene | hold weight w | target | gamma | PSNR | SSIM | LPIPS | DreamSim | held | tint (x floor) | outside metrics beyond seed noise | adversarial |",
         "|:--|--:|:--|--:|--:|--:|--:|--:|:--|--:|:--|:--|"]
    for r in df.itertuples():
        if pd.isna(getattr(r, "flip", np.nan)):
            L.append(f"| {r.scene.split('/')[1]} | {r.wdefault:g} | {r.target} | {r.gamma:g} | {r.psnr:+.2f} | {r.ssim:+.2f} | {r.lpips_vgg:+.2f} | {r.dreamsim:+.2f} | "
                     f"{'yes' if r.held else 'no'} | {r.tint:.2f} | (not computed yet) | {'YES' if r.adversarial else 'no'} |")
            continue
        if not hasattr(r, "psnr") or pd.isna(getattr(r, "psnr", np.nan)):
            L.append(f"| {r.scene} | {r.wdefault:g} | {r.target} | {r.gamma:g} | (metrics not computed) | | | | | {r.tint:.2f} | | |")
            continue
        out = ", ".join(f"{m} {getattr(r, m):+.1f}" for m in OUTSIDE if getattr(r, m) > 1) or "none"
        L.append(f"| {r.scene.split('/')[1]}{' [+' + r.hold_extra + ' held]' if r.hold_extra else ''} | {r.wdefault:g} | {r.target} | {r.gamma:g} | {r.psnr:+.2f} | {r.ssim:+.2f} | {r.lpips_vgg:+.2f} | {r.dreamsim:+.2f} | "
                 f"{'yes' if r.held else 'no'} | {r.tint:.2f} | {out} | {'YES' if r.adversarial else 'no'} |")
    open(os.path.join(P.S6_REC, "results", "ext_table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
if __name__ == "__main__":
    main()
