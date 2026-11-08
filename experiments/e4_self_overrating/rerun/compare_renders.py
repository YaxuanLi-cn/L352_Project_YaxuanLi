#!/usr/bin/env python
import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import helpers as C              
import numpy as np              
import pandas as pd              
from PIL import Image              
OLD = P.ORIG_STIMULI
NEW = P.RERUN_STIMULI
RUNS = P.RERUN_RUNS


def stats(x, y):
    d = np.abs(x.astype(np.int16) - y.astype(np.int16))
    mse = float((d.astype(np.float64) ** 2).mean())
    return int(d.max()), (10 * np.log10(255 ** 2 / mse) if mse > 0 else np.inf)


def decode(path):
    w, h = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                    "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h * w * 3 // 2)


def one(job):
    scene, model = job
    old, new = os.path.join(OLD, scene, model), os.path.join(NEW, scene, "s0", model)
    rows = []
    if os.path.exists(os.path.join(old, "video.mp4")):
        a, b = decode(os.path.join(old, "video.mp4")), decode(os.path.join(new, "video.mp4"))
        rows = [stats(x, y) for x, y in zip(a, b)]
        kind = "video frames (yuv420p)"
    else:
        fa, fb = sorted(glob.glob(os.path.join(old, "*.png"))), sorted(glob.glob(os.path.join(new, "test", "*.png")))
        rows = [stats(np.asarray(Image.open(x).convert("RGB")), np.asarray(Image.open(y).convert("RGB"))) for x, y in zip(fa, fb)]
        kind = "png"
    mx, ps = [r[0] for r in rows], np.array([r[1] for r in rows])
    return dict(scene=scene, model=model, kind=kind, views=len(rows), identical_views=int(sum(m == 0 for m in mx)),
                max_abs_diff=int(max(mx)), psnr_min=float(ps.min()), psnr_median=float(np.median(ps)), note="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=P.RERUN_COMPARE)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    jobs = []
    for scene in sorted(os.listdir(OLD)):
        if not os.path.isdir(os.path.join(OLD, scene)):
            continue
        for model in sorted(os.listdir(os.path.join(OLD, scene))):
            if model == "reference":
                continue
            jobs.append((scene, model))
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = [r for r in ex.map(one, jobs)]
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "render_vs_paper.csv"), index=False)
    old_runs = P.ORIG_RUNS
    mrows = []
    for scene, model in jobs:
        fa, fb = (os.path.join(r, scene, "s0", model, "metrics.csv") for r in (old_runs, RUNS))
        a, b = (pd.read_csv(f) for f in (fa, fb))
        a, b = (x[x.split == "test"].pivot(index="view_id", columns="metric", values="value") for x in (a, b))
        for met in ("psnr", "lpips_vgg", "dists", "cvvdp_image"):
            d = (b[met] - a[met]).dropna()
            big = d.index[d.abs() > 0.5] if met == "psnr" else []
            mrows.append(dict(scene=scene, model=model, metric=met, views=len(d), mean_paper=float(a[met].mean()),
                              mean_new=float(b[met].mean()), diff_of_means=float(b[met].mean() - a[met].mean()),
                              median_abs_diff_per_view=float(d.abs().median()), max_abs_diff_per_view=float(d.abs().max()),
                              views_moved_gt_0p5=len(big), first_moved=int(min(big)) if len(big) else -1,
                              last_moved=int(max(big)) if len(big) else -1))
    pd.DataFrame(mrows).to_csv(os.path.join(args.out, "metrics_vs_paper_by_scene.csv"), index=False)
    base = df[df.model == "none__base"]
    C.log(f"{len(df)} models compared; base: identical views {int(base.identical_views.sum())} / {int(base.views.sum())} in {len(base)} scenes")
    return df
if __name__ == "__main__":
    main()
