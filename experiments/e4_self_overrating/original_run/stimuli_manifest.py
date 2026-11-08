#!/usr/bin/env python
import argparse
import glob
import os
import shutil
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import pandas as pd              
import yaml              
from nvsqa import datasets, pipeline, video_io              
from gh import evaluate as E              


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = os.path.normpath(os.path.join(P.ROOT, cfg["stimuli_root"]))
    rows = []
    for spec in cfg["scenes"]:
        sdir = os.path.join(root, spec.replace("/", "_"))
        ref_dir = os.path.join(sdir, "reference")
        os.makedirs(ref_dir, exist_ok=True)
        scene = datasets.load_scene(spec)
        names = [scene.names[i] for i in scene.test_ids]
        if scene.fps > 0:
            dst = os.path.join(ref_dir, "video.mp4")
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(pipeline.PART1_ROOT, spec, "reference.mp4"), dst)
        elif len(glob.glob(os.path.join(ref_dir, "*.png"))) != len(names):
            E.write_pngs(ref_dir, E.crop_ref(scene, scene.test_ids))
        for d in sorted(glob.glob(os.path.join(sdir, "*"))):
            cond = os.path.basename(d)
            loss, variant = ("reference", "reference") if cond == "reference" else cond.split("__")
            if scene.fps > 0:
                f = os.path.join(d, "video.mp4")
                info = video_io.probe(f)
                rows.append(dict(scene=spec, loss=loss, variant=variant, seed=0, kind="video",
                                 path=os.path.relpath(f, root), view_id=-1, source_view="", n_frames=info["nb_frames"],
                                 fps=info["fps"], width=info["width"], height=info["height"],
                                 encoding="H.264 lossless (qp 0), yuv420p"))
            else:
                for j, n in enumerate(names):
                    f = os.path.join(d, f"{j:03d}.png")
                    rows.append(dict(scene=spec, loss=loss, variant=variant, seed=0, kind="png",
                                     path=os.path.relpath(f, root), view_id=j, source_view=n, n_frames=1,
                                     fps=0, width=scene.width, height=scene.height, encoding="PNG (lossless, 8 bit)"))
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(root, "manifest.csv"), index=False)
    print(f"manifest: {len(m)} rows -> {os.path.join(root, 'manifest.csv')}")
if __name__ == "__main__":
    main()
