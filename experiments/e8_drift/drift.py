#!/usr/bin/env python
import argparse
import glob
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import helpers as C              
import uncertainty as U              
import numpy as np              
import pandas as pd              
from gh import metrics as M              
from nvsqa import video_io              
CFG = P.code(P.E7, "configs", "pilot.yaml")
PAPER = P.ORIG_STIMULI
RERUN = P.RERUN_STIMULI
OUT = P.DRIFT
CTRL, BASE = "default__ctrl", "none__base"
IMG = list(M.IMAGE_METRICS)
KINDS = ["drift", "change_paper", "change_rerun", "seed"]
SEED_SCENE = "gsqa/truck"
GAIN_MODELS = (CTRL, "cvvdp__pure", "cvvdp__mixed")


def loss_models():
    import yaml
    cfg = yaml.safe_load(open(P.code(P.E4, "original_run", "configs", "full.yaml")))
    return [f"{l}__pure" for l in cfg["pure"]] + [f"{l}__mixed" for l in cfg["mixed"]]


def held_out(spec):
    import json
    return json.load(open(os.path.join(P.RERUN_VIEWS, f"{C.scene_dir(spec)}.json")))["heldout"]
_video = {}


def frames(source, spec, model, views, seed=0):
    d = os.path.join(PAPER, C.scene_dir(spec), model) if source == "paper" else \
        os.path.join(RERUN, C.scene_dir(spec), f"s{seed}", model)
    if spec.startswith("gsqa/"):
        v = os.path.join(d, "video.mp4")
        if v not in _video:
            _video.clear()
            _video[v] = video_io.read_video(v)[0]
        return np.ascontiguousarray(_video[v][views])
    files = sorted(glob.glob(os.path.join(d, "*.png" if source == "paper" else os.path.join("test", "*.png"))))
    return C.read_pngs([files[v] for v in views])


def jobs_of(spec, kind):
    lm = loss_models()
    if kind == "drift":
        return [(c, ("paper", c, 0), ("rerun", c, 0)) for c in [BASE, CTRL] + lm]
    if kind == "change_paper":
        return [(c, ("paper", c, 0), ("paper", CTRL, 0)) for c in lm]
    if kind == "change_rerun":
        return [(c, ("rerun", c, 0), ("rerun", CTRL, 0)) for c in lm]
    if kind == "seed":
        return [(c, ("rerun", c, 0), ("rerun", c, s)) for c in [CTRL] + lm for s in (1, 2)] if spec == SEED_SCENE else []


def step_pairs(args):
    cfg = C.load_cfg(CFG)
    m = None
    os.makedirs(os.path.join(OUT, "pairs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)
    for kind in args.kinds:
        for spec in args.scenes or cfg["scenes"]:
            jobs = jobs_of(spec, kind)
            f = os.path.join(OUT, "pairs", f"{C.scene_dir(spec)}__{kind}.parquet")
            lock = f + ".lock"
            if not jobs or os.path.exists(f) or not C.acquire(lock):
                continue
            if m is None:
                m = M.Metrics()
            views = held_out(spec)[:args.max_views] if args.max_views else held_out(spec)
            rows, maps = [], []
            for model, a, b in jobs:
                A, B = frames(a[0], spec, a[1], views, a[2]), frames(b[0], spec, b[1], views, b[2])
                for direction, (t, r) in (("ab", (A, B)), ("ba", (B, A))):
                    vals = m.evaluate(t, r, args.metrics)
                    for k, v in vals.items():
                        rows += [dict(scene=spec, kind=kind, model=model, a=f"{a[0]}:s{a[2]}", b=f"{b[0]}:s{b[2]}",
                                      view=int(vw), direction=direction, metric=k, value=float(x)) for vw, x in zip(views, v)]
                if kind == "drift":
                    for vw, x, y in zip(views, A, B):
                        sq = ((x.astype(np.float64) - y.astype(np.float64)) / 255.0) ** 2
                        d = np.sqrt(U.tile_view(sq.mean(axis=2), cfg["tile"]).mean(axis=(1, 2)))
                        maps += [dict(scene=spec, model=model, view=int(vw), tile=i, D=float(t_),
                                      rms_image=float(np.sqrt(sq.mean()))) for i, t_ in enumerate(d)]
                C.log(f"{spec} {kind} {model} done")
            if maps and not args.max_views:
                pd.DataFrame(maps).to_parquet(os.path.join(OUT, "maps", f"{C.scene_dir(spec)}.parquet"))
            if args.max_views:
                pd.DataFrame(rows).to_parquet(os.path.join(args.test_out, os.path.basename(f)))
            else:
                pd.DataFrame(rows).to_parquet(f)
            C.release(lock)


def colour_match(x, y, mode):
    X = x.reshape(-1, 3).astype(np.float64) / 255.0
    Y = y.reshape(-1, 3).astype(np.float64) / 255.0
    if mode == "affine":
        X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        W = np.linalg.solve(X1.T @ X1, X1.T @ Y)
        out = X1 @ W
    elif mode == "gain":
        out = np.empty_like(X)
        for c in range(3):
            a, b = np.polyfit(X[:, c], Y[:, c], 1)
            out[:, c] = a * X[:, c] + b
    return np.round(np.clip(out, 0, 1) * 255.0).astype(np.uint8).reshape(x.shape)


def step_colour(args):
    cfg = C.load_cfg(CFG)
    m = None
    os.makedirs(os.path.join(OUT, "colour"), exist_ok=True)
    for spec in args.scenes or cfg["scenes"]:
        f = os.path.join(OUT, "colour", f"{C.scene_dir(spec)}.parquet")
        lock = f + ".lock"
        if os.path.exists(f) or not C.acquire(lock):
            continue
        if m is None:
            m = M.Metrics()
        views = held_out(spec)[:args.max_views] if args.max_views else held_out(spec)
        sd = C.scene_dir(spec)
        rf = sorted(glob.glob(os.path.join(RERUN, sd, "reference", "test", "*.png")))
        R = C.read_pngs([rf[v] for v in views])
        rows = []
        for model in [CTRL] + loss_models():
            tf = sorted(glob.glob(os.path.join(RERUN, sd, "s0", model, "test", "*.png")))
            T = C.read_pngs([tf[v] for v in views])
            for variant in ("none", "affine", "gain") if model in GAIN_MODELS else ("none", "affine"):
                X = T if variant == "none" else np.stack([colour_match(t, r, variant) for t, r in zip(T, R)])
                vals = m.evaluate(X, R, args.metrics)
                for k, v in vals.items():
                    rows += [dict(scene=spec, model=model, view=int(vw), variant=variant, metric=k, value=float(x))
                             for vw, x in zip(views, v)]
            C.log(f"{spec} colour {model} done")
        pd.DataFrame(rows).to_parquet(os.path.join(args.test_out, os.path.basename(f)) if args.max_views else f)
        C.release(lock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pairs", "colour"])
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--kinds", nargs="*", default=KINDS)
    ap.add_argument("--metrics", nargs="*", default=IMG)
    ap.add_argument("--max-views", type=int, default=0, help="test run on the first views; output goes to --test-out")
    ap.add_argument("--test-out", default="")
    args = ap.parse_args()
    if args.max_views:
        os.makedirs(args.test_out, exist_ok=True)
    {"pairs": step_pairs, "colour": step_colour}[args.step](args)
if __name__ == "__main__":
    main()
