#!/usr/bin/env python
import os
import sys
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import pandas as pd              
from nvsqa.metrics import SIGN              
METRICS = ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg", "flip", "cvvdp_image", "cvvdp"]
pd.set_option("display.width", 250)


def flicker():
    f = os.path.join(P.FLICKER_RESULTS, "temporal_probe.csv")
    t = pd.read_csv(f)
    metrics = METRICS + ["cvvdp_sust"]
    rows = []
    for key, g in t.groupby(["scene", "method", "condition"]):
        a, b = g[g.variant == "A"].iloc[0], g[g.variant == "B"].iloc[0]
        avg = g[g.variant == "avg"]
        for _, r in g[g.variant.str.startswith("mix")].iterrows():
            row = dict(scene=key[0], method=key[1], condition=key[2], kind=a.kind, period=int(r.period),
                       hz=r.fps / (2 * r.period), psnr_parent=(a.psnr + b.psnr) / 2,
                       cvvdp_parent=(a.cvvdp + b.cvvdp) / 2, seed_gap_jod=abs(a.cvvdp - b.cvvdp),
                       temp_ratio_parent=(a.temp_ratio + b.temp_ratio) / 2, temp_ratio_mix=r.temp_ratio)
            for m in metrics:
                row["d_" + m] = SIGN[m] * (r[m] - (a[m] + b[m]) / 2)
            for m in ("psnr", "cvvdp", "cvvdp_sust"):
                row["avg_d_" + m] = avg.iloc[0][m] - (a[m] + b[m]) / 2
            rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(P.FLICKER_RESULTS, "temporal_probe_summary.csv"), index=False)
    print("\n=== flicker probe: change of the (oriented) score, mixture minus mean of its two parents ===")
    cols = ["d_" + m for m in metrics]
    show = s.groupby(["method", "condition", "period"])[cols + ["cvvdp_parent", "temp_ratio_parent", "temp_ratio_mix"]].mean()
    print(show.round(3).to_string())
    print("\n=== mean over all conditions, by switching period ===")
    print(s.groupby("period")[cols].mean().round(4).to_string())
    print("\nreference scale: mean |JOD difference between the two seeds| = %.3f" % s.seed_gap_jod.mean())
    one = s[s.period == 1]
    print("\n=== control: pixel-wise average of the two videos vs the mixture that switches every frame (%d conditions) ===" % len(one))
    print("   sustained channels: average %+.3f JOD, mixture %+.3f JOD, mean |difference| %.3f" %
          (one.avg_d_cvvdp_sust.mean(), one.d_cvvdp_sust.mean(), (one.avg_d_cvvdp_sust - one.d_cvvdp_sust).abs().mean()))
    print("   video mode:         average %+.3f JOD, mixture %+.3f JOD, mean |difference| %.3f" %
          (one.avg_d_cvvdp.mean(), one.d_cvvdp.mean(), (one.avg_d_cvvdp - one.d_cvvdp).abs().mean()))
    print("   PSNR of the average: %+.2f dB against the parents" % one.avg_d_psnr.mean())
if __name__ == "__main__":
    flicker()
