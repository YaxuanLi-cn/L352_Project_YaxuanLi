#!/usr/bin/env python
import argparse
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import cv2              
import matplotlib              
matplotlib.use("Agg")
import matplotlib.pyplot as plt              
import numpy as np              
import pandas as pd              
import yaml              
from gh import metrics as M              
SHOW = [("reference", "reference", "reference"), ("default", "ctrl", "ctrl (default loss)"),
        ("lpips_vgg", "mixed", "LPIPS-VGG (mixed)"), ("lpips_alex", "pure", "LPIPS-Alex (pure)"),
        ("dists", "pure", "DISTS (pure)"), ("cvvdp", "pure", "CVVDP (pure)")]
SIZE = 144


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--scene", default="mip360/garden")
    ap.add_argument("--view", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = os.path.normpath(os.path.join(P.ROOT, cfg["stimuli_root"]))
    R = os.path.normpath(os.path.join(P.ROOT, cfg["results_dir"]))
    df = pd.read_csv(os.path.join(R, "metrics.csv"))
    df = df[(df.scene == args.scene) & (df.seed == 0) & (df.split == "test")]
    if args.view is None:
        p = df[(df.variant == "ctrl") & (df.metric == "psnr")].sort_values("value")
        args.view = int(p.view_id.iloc[len(p) // 2])
    sdir = os.path.join(root, args.scene.replace("/", "_"))
    def load(loss, var):
        d = "reference" if loss == "reference" else f"{loss}__{var}"
        return cv2.cvtColor(cv2.imread(os.path.join(sdir, d, f"{args.view:03d}.png")), cv2.COLOR_BGR2RGB)
    ref = load("reference", "")
    lum = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(lum, (SIZE, SIZE))
    sd = np.sqrt(np.maximum(cv2.blur(lum * lum, (SIZE, SIZE)) - mean ** 2, 0))
    h = SIZE // 2
    inner = sd[h:-h, h:-h]
    crops = []
    for pick in (np.argmax, np.argmin):
        y, x = np.unravel_index(pick(inner), inner.shape)
        crops.append((y, x))
    fig, axes = plt.subplots(len(crops), len(SHOW), figsize=(1.8 * len(SHOW), 1.8 * len(crops) + 0.9),
                             gridspec_kw=dict(hspace=0.03, wspace=0.04))
    for ci, (loss, var, name) in enumerate(SHOW):
        img = load(loss, var)
        for ri, (y, x) in enumerate(crops):
            ax = axes[ri, ci]
            ax.imshow(img[y:y + SIZE, x:x + SIZE], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if ri == 0:
                ax.set_title(name, fontsize=8)
        if loss != "reference":
            q = df[(df.loss == loss) & (df.variant == var) & (df.view_id == args.view)].set_index("metric").value
            axes[-1, ci].set_xlabel(f"PSNR {q['psnr']:.2f} dB\nLPIPS-Alex {q['lpips_alex']:.3f}\nDISTS {q['dists']:.3f}\n"
                                    f"CVVDP {q['cvvdp_image']:.2f} JOD", fontsize=7)
    axes[0, 0].set_ylabel("most textured crop", fontsize=7)
    axes[1, 0].set_ylabel("smoothest crop", fontsize=7)
    fig.suptitle(f"{args.scene}, test view {args.view} (metric values of the whole view)", fontsize=9, x=0.12, y=0.99,
                 ha="left")
    out = os.path.join(R, "figures", f"examples_{args.scene.replace('/', '_')}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("written", out)
if __name__ == "__main__":
    main()
