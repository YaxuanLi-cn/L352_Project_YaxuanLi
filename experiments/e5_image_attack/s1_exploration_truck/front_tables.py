#!/usr/bin/env python
import os
import sys
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import pandas as pd              
P90 = dict(psnr=0.45, ssim=0.008, lpips_vgg=0.011)
d = pd.read_csv(P.rec(P.E5, P.S1, "mad_summary_gsqa_truck.csv"))
d = d[d.run.str.startswith("front-")].copy()
m = d.run.str.extract(r"front-(?P<anchor>\w+?)-(?P<target>\w+)-(?P<gamma>[0-9.]+)__")
d = pd.concat([d, m], axis=1)
d["gamma"] = d.gamma.astype(float)
NAME = {"psnr": "只守 PSNR", "default": "守 PSNR + SSIM", "triplet": "守 PSNR + SSIM + LPIPS"}
target = sys.argv[1] if len(sys.argv) > 1 else "dists"
print("| 损失里守住的（= 论文里报告的） | 推力 γ | ΔPSNR | ΔSSIM | ΔLPIPS-VGG | DISTS | DreamSim | LPIPS-Alex | ColorVideoVDP | FSIM | VIF | FLIP |")
print("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for a in ("psnr", "default", "triplet"):
    for _, r in d[(d.anchor == a) & (d.target == target)].sort_values("gamma").iterrows():
        print(f"| {NAME[a]} | {r.gamma:g} | {r.d_psnr:+.2f} dB（{abs(r.d_psnr) / P90['psnr']:.1f}） | {r.d_ssim:+.3f}（{abs(r.d_ssim) / P90['ssim']:.1f}） | "
              f"{r.d_lpips_vgg:+.3f}（{abs(r.d_lpips_vgg) / P90['lpips_vgg']:.1f}） | {r.z_dists:+.1f} | {r.z_dreamsim:+.1f} | {r.z_lpips_alex:+.1f} | "
              f"{r.d_cvvdp_image:+.2f} JOD | {r.z_fsim:+.1f} | {r.z_vif:+.1f} | {r.z_flip:+.1f} |")
