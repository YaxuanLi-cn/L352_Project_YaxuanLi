#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys
import time
os.environ.setdefault("OMP_NUM_THREADS", "4")
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import cv2              
import numpy as np              
from nvsqa import datasets, pipeline, video_io              
from nvsqa.metrics import MetricSuite              


def pending(root, name):
    out = []
    for meta in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*", "meta.json"))):
        d = os.path.dirname(meta)
        if not os.path.exists(os.path.join(d, name)):
            out.append(d)
    return out


def load_pngs(d):
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    return np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--display", default="fhd_3h")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on the condition path")
    ap.add_argument("--flip-procs", type=int, default=5, help="worker processes for FLIP (2 threads each)")
    args = ap.parse_args()
    name = "metrics.json" if args.display == "fhd_3h" else f"metrics_{args.display}.json"
    suite = MetricSuite(display=args.display, flip_procs=args.flip_procs)
    refs = {}
    def reference(spec, is_video):
        if spec not in refs:
            refs.clear()
            if is_video:
                refs[spec] = video_io.read_video(os.path.join(pipeline.PART1_ROOT, spec, "reference.mp4"))[0]
            else:
                scene = datasets.load_scene(spec)
                refs[spec] = pipeline.reference_frames(scene).numpy()
        return refs[spec]
    while True:
        todo = [d for d in pending(pipeline.PART1_ROOT, name) if args.only in d]
        for d in todo:
            t0 = time.time()
            meta = json.load(open(os.path.join(d, "meta.json")))
            spec = meta["scene"]
            is_video = meta["fps"] > 0
            ref = reference(spec, is_video)
            if is_video:
                test = video_io.read_video(os.path.join(d, "video.mp4"))[0]
                scores, frames = suite.video(test, ref, meta["fps"])
            else:
                test = load_pngs(os.path.join(d, "renders"))
                frames = suite.images(test, ref)
                scores = {k: float(np.mean(v)) for k, v in frames.items()}
            res = dict(display=args.display, ppd=suite.ppd, scores=scores,
                       frames={k: np.asarray(v).tolist() for k, v in frames.items()})
            with open(os.path.join(d, name), "w") as f:
                json.dump(res, f)
            print(f"{time.strftime('%m-%d %H:%M:%S')} {os.path.relpath(d, pipeline.PART1_ROOT)}: " +
                  " ".join(f"{k}={v:.4g}" for k, v in scores.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
        if not args.loop:
            break
        locks = glob.glob(os.path.join(pipeline.PART1_ROOT, ".locks", "*"))
        if not todo and not locks:
            break
        if not todo:
            time.sleep(60)
    suite.close()
if __name__ == "__main__":
    main()
