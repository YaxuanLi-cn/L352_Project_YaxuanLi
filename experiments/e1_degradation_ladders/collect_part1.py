#!/usr/bin/env python
import glob
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
from nvsqa import pipeline              


def level_of(meta):
    if meta["kind"] == "full":
        return "iterations", meta["step"]
    if meta["kind"] == "views":
        return "views", meta["num_train_views"]
    return "gaussians", meta["value"]


def main(name="metrics.json", out_suffix=""):
    rows, frows = [], []
    for mfile in sorted(glob.glob(os.path.join(pipeline.PART1_ROOT, "*", "*", "*", "*", "meta.json"))):
        d = os.path.dirname(mfile)
        meta = json.load(open(mfile))
        res = json.load(open(os.path.join(d, name)))
        ladder, level = level_of(meta)
        dataset, scene = meta["scene"].split("/")
        base = dict(dataset=dataset, scene=scene, method=meta["method"], condition=meta["condition"],
                    ladder=ladder, level=level, seed=meta["seed"], step=meta["step"],
                    num_train_views=meta["num_train_views"], num_train_views_full=meta["num_train_views_full"],
                    num_gaussians=meta.get("num_gaussians", np.nan), train_time=meta["train_time"],
                    is_video=meta["fps"] > 0, fps=meta["fps"])
        rows.append(dict(base, **res["scores"]))
        per_frame = {k: v for k, v in res["frames"].items() if np.ndim(v) == 1}
        n = len(next(iter(per_frame.values())))
        for i in range(n):
            frows.append(dict(dataset=dataset, scene=scene, method=meta["method"], condition=meta["condition"],
                              ladder=ladder, level=level, seed=meta["seed"], frame=i,
                              **{k: v[i] for k, v in per_frame.items()}))
    out = P.PART1_RESULTS + out_suffix
    os.makedirs(out, exist_ok=True)
    cond = pd.DataFrame(rows)
    cond.to_csv(os.path.join(out, "conditions.csv"), index=False)
    pd.DataFrame(frows).to_csv(os.path.join(out, "frames.csv"), index=False)
    print(f"{len(cond)} conditions, {len(frows)} frames -> {out}")
    print(cond.groupby(["dataset", "scene", "method", "ladder"]).size().unstack(fill_value=0))
if __name__ == "__main__":
    main()
