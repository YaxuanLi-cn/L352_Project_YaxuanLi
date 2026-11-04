#!/usr/bin/env python
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
from PIL import Image              
from nvsqa import datasets, descriptors, pipeline, video_io              


def main():
    rows, refs = [], {}
    metas = sorted(glob.glob(os.path.join(pipeline.PART1_ROOT, "*", "*", "*", "*", "meta.json")))
    for mfile in metas:
        d = os.path.dirname(mfile)
        meta = json.load(open(mfile))
        if meta["seed"] != 42:
            continue
        cache = os.path.join(d, "descriptors.json")
        if os.path.exists(cache):
            desc = json.load(open(cache))
        else:
            scene = meta["scene"]
            is_video = meta["fps"] > 0
            if scene not in refs:
                refs.clear()
                if is_video:
                    refs[scene] = video_io.read_video(os.path.join(pipeline.PART1_ROOT, scene, "reference.mp4"))[0]
                else:
                    refs[scene] = np.asarray(pipeline.reference_frames(datasets.load_scene(scene)))
            if is_video:
                test = video_io.read_video(os.path.join(d, "video.mp4"))[0]
            else:
                names = sorted(os.listdir(os.path.join(d, "renders")))
                test = np.stack([np.asarray(Image.open(os.path.join(d, "renders", n)).convert("RGB")) for n in names])
            desc = descriptors.describe(test, refs[scene], flow_every=10 ** 6, chunk=8)
            if not is_video:
                desc["temp_ratio"] = desc["temp_err"] = float("nan")
            desc.pop("shift_px", None)
            with open(cache, "w") as f:
                json.dump(desc, f)
            print(scene, meta["method"], meta["condition"], " ".join(f"{k}={v:.3f}" for k, v in desc.items()), flush=True)
        dataset, sc = meta["scene"].split("/")
        rows.append(dict(dataset=dataset, scene=sc, method=meta["method"], condition=meta["condition"], **desc))
    out = os.path.join(P.PART1_RESULTS, "descriptors.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} conditions -> {out}")
if __name__ == "__main__":
    main()
