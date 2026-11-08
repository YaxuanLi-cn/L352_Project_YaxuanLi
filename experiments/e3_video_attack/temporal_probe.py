#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import numpy as np              
import pandas as pd              
from nvsqa import descriptors, pipeline, video_io              
from nvsqa.metrics import IMAGE_METRICS, MetricSuite              
PERIODS = [1, 2, 4, 8, 15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None, help="e.g. gsqa/truck (default: all with seed pairs)")
    ap.add_argument("--conditions", nargs="*", default=None, help="restrict to these condition names")
    ap.add_argument("--out", default=None, help="write to this CSV instead (to run several scenes in parallel; "
                                                "concatenate the files into temporal_probe.csv afterwards)")
    args = ap.parse_args()
    out_csv = args.out or os.path.join(P.FLICKER_RESULTS, "temporal_probe.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    rows = done.to_dict("records")
    seen = set(zip(done.get("scene", []), done.get("method", []), done.get("condition", []), done.get("variant", [])))
    variants = ["A", "B"] + [f"mix{k}" for k in PERIODS] + ["avg"]
    suite = None
    for b_dir in sorted(glob.glob(os.path.join(pipeline.PART1_ROOT, "gsqa", "*", "*", "*-s1"))):
        a_dir = b_dir[:-3]
        meta = json.load(open(os.path.join(a_dir, "meta.json")))
        key = (meta["scene"], meta["method"], meta["condition"])
        todo = [v for v in variants if key + (v,) not in seen]
        if not todo or (args.scenes and meta["scene"] not in args.scenes) or \
                (args.conditions and meta["condition"] not in args.conditions):
            continue
        ma, mb = (json.load(open(os.path.join(d, "metrics.json"))) for d in (a_dir, b_dir))
        ref = video_io.read_video(os.path.join(pipeline.PART1_ROOT, meta["scene"], "reference.mp4"))[0]
        a = video_io.read_video(os.path.join(a_dir, "video.mp4"))[0]
        b = video_io.read_video(os.path.join(b_dir, "video.mp4"))[0]
        fps = meta["fps"]
        t = np.arange(len(a))
        if suite is None:
            suite = MetricSuite(metrics=["cvvdp", "cvvdp_sust"], flip_procs=0)
        for name in todo:
            period = 0
            if name in ("A", "B"):
                vid = a if name == "A" else b
                scores = dict((ma if name == "A" else mb)["scores"])
            elif name == "avg":
                vid = ((a.astype(np.uint16) + b.astype(np.uint16) + 1) // 2).astype(np.uint8)
                scores, _ = suite.video(vid, ref, fps, names=["psnr", "ssim", "cvvdp", "cvvdp_sust"])
            else:
                period = int(name[3:])
                from_a = (t // period) % 2 == 0
                vid = np.where(from_a[:, None, None, None], a, b)
                scores, _ = suite.video(vid, ref, fps, names=["cvvdp", "cvvdp_sust"], image_metrics=False)
                for m in IMAGE_METRICS:
                    scores[m] = float(np.where(from_a, ma["frames"][m], mb["frames"][m]).mean())
            d = descriptors.describe(vid, ref, flow_every=10 ** 6)
            rows.append(dict(scene=meta["scene"], method=meta["method"], condition=meta["condition"],
                             kind=meta["kind"], variant=name, period=period, fps=fps,
                             temp_ratio=d["temp_ratio"], temp_err=d["temp_err"], **scores))
            print(key, name, " ".join(f"{m}={v:.4g}" for m, v in scores.items()), flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    if suite is not None:
        suite.close()
    print("wrote", out_csv)
if __name__ == "__main__":
    main()
