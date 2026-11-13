#!/usr/bin/env python
import glob
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
sys.path.insert(0, HERE)
import run_suff as S              
import numpy as np              
import pandas as pd              
R = S.R
RERUN = P.RERUN_STIMULI
SCENES = ["gsqa/truck", "mip360/bonsai", "mip360/garden"]


def held_views(sdn, n=20):
    views = json.load(open(os.path.join(P.RERUN_VIEWS, f"{sdn}.json")))
    held = np.asarray(views["heldout"])
    return held[np.unique(np.round(np.linspace(0, len(held) - 1, n)).astype(int))]


def table(metrics, scene, name, d, held, ref):
    files = [os.path.join(d, f"{int(v):03d}.png") for v in held]
    img = R.C.read_pngs(files)
    rows = []
    for (res, m), v in S.evaluate_both(metrics, img, ref).items():
        rows += [dict(scene=scene, name=name, split="test", view=int(vw), res=res, metric=m, value=float(x)) for vw, x in zip(held, v)]
    return pd.DataFrame(rows)


def main():
    scenes = sys.argv[1:] or SCENES
    metrics = R.X.Metrics()
    os.makedirs(P.S2_AUX, exist_ok=True)
    for scene in scenes:
        sdn = scene.replace("/", "_")
        held = held_views(sdn)
        ref = R.C.read_pngs([os.path.join(RERUN, sdn, "reference", "test", f"{int(v):03d}.png") for v in held])
        todo = [("control", os.path.join(RERUN, sdn, "s0", "default__ctrl", "test"))]
        if scene == "gsqa/truck":
            for d in sorted(glob.glob(os.path.join(P.S1_RUNS, "gsqa_truck", "front-*"))):
                if os.path.exists(os.path.join(d, "done.json")):
                    todo.append((os.path.basename(d), os.path.join(d, "test")))
        for name, d in todo:
            out = os.path.join(P.S2_AUX, f"{sdn}__{name}.parquet")
            if os.path.exists(out):
                continue
            t = table(metrics, scene, name, d, held, ref)
            t.to_parquet(out)
            f = t[t.res == "full"].groupby("metric").value.mean()
            h = t[t.res == "half"].groupby("metric").value.mean()
            R.log(f"{scene} {name}: psnr {f['psnr']:.3f}/{h['psnr']:.3f} dists {f['dists']:.4f}/{h['dists']:.4f} dreamsim {f['dreamsim']:.4f}/{h['dreamsim']:.4f} (full/half)")
if __name__ == "__main__":
    main()
