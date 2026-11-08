#!/usr/bin/env python
import argparse
import functools
import glob
import os
import sys
import time
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
ROOT = P.ROOT
import cv2              
import numpy as np              
import pandas as pd              
import torch              
import yaml              
from nvsqa import datasets, pipeline, video_io              
from nvsqa import gs as nvsqa_gs              
from gh import evaluate as E              
from gh import metrics as M              
@torch.no_grad()


def convention_metrics(test_u8, ref_u8, lpips_vgg):
    out = {"lpips_vgg_01input": [], "ssim_padded": []}
    for t, r in zip(test_u8, ref_u8):
        t = torch.from_numpy(np.ascontiguousarray(t)).cuda().permute(2, 0, 1)[None].float() / 255
        r = torch.from_numpy(np.ascontiguousarray(r)).cuda().permute(2, 0, 1)[None].float() / 255
        out["lpips_vgg_01input"].append(float(lpips_vgg(t, r, normalize=False)))
        out["ssim_padded"].append(float(nvsqa_gs.ssim_map(t, r).mean()))
    return {k: np.asarray(v) for k, v in out.items()}


def rows_from(vals, key, split="test"):
    return [dict(key, split=split, view_id=j, metric=m, value=float(x)) for m, v in vals.items() for j, x in enumerate(v)]


def cmd_conventions(args):
    cfg = yaml.safe_load(open(args.config))
    metrics = M.Metrics()
    net = metrics._net("lpips_vgg")
    for spec in cfg["scenes"]:
        if not spec.startswith("mip360/"):
            sdir = os.path.join(ROOT, cfg["stimuli_root"], spec.replace("/", "_"))
            test = video_io.read_video(os.path.join(sdir, "none__base", "video.mp4"))[0].astype(np.float64)
            ref = video_io.read_video(os.path.join(pipeline.PART1_ROOT, spec, "reference.mp4"))[0].astype(np.float64)
            psnr = -10 * np.log10(((test - ref) ** 2).mean(axis=(1, 2, 3)) / 255.0 ** 2)
            key = dict(scene=spec, loss="none", variant="base", seed=0)
            out = os.path.join(ROOT, cfg["runs_root"], spec.replace("/", "_"), "s0", "none__base", "extra_metrics.csv")
            pd.DataFrame(rows_from({"psnr_decoded": psnr}, key, split="path")).to_csv(out, index=False)
            print(spec, "psnr_decoded", round(float(psnr.mean()), 3), "->", out, flush=True)
            continue
        d = os.path.join(ROOT, cfg["stimuli_root"], spec.replace("/", "_"), "none__base")
        files = sorted(glob.glob(os.path.join(d, "*.png")))
        scene = datasets.load_scene(spec)
        ref = E.crop_ref(scene, scene.test_ids)
        test = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files])
        vals = convention_metrics(test, ref, net)
        key = dict(scene=spec, loss="none", variant="base", seed=0)
        out = os.path.join(ROOT, cfg["runs_root"], spec.replace("/", "_"), "s0", "none__base", "extra_metrics.csv")
        pd.DataFrame(rows_from(vals, key)).to_csv(out, index=False)
        print(spec, {k: round(float(v.mean()), 4) for k, v in vals.items()}, "->", out, flush=True)


def cmd_diag(args):
    name = args.name or f"{args.scene.replace('/', '_')}_grow{args.grow_grad2d:g}_seed{42 + args.seed}"
    out_dir = os.path.join(P.ORIG, "runs_diag", name)
    if os.path.exists(os.path.join(out_dir, "metrics.csv")):
        print("already done:", out_dir)
        return
    os.makedirs(out_dir, exist_ok=True)
    scene = datasets.load_scene(args.scene)
    from gsplat.strategy import DefaultStrategy
    nvsqa_gs.DefaultStrategy = functools.partial(DefaultStrategy, grow_grad2d=args.grow_grad2d)
    cfg = nvsqa_gs.GSConfig(seed=42 + args.seed, max_steps=30_000)
    def log(msg):
        print(f"{time.strftime('%m-%d %H:%M:%S')} [{name}] {msg}", flush=True)
    model, info = nvsqa_gs.train_gs(scene, scene.train_ids, cfg, device="cuda", log=log, log_every=5000)
    metrics = M.Metrics()
    test = E.render_views(model, scene, scene.test_ids)
    ref = E.crop_ref(scene, scene.test_ids)
    key = dict(scene=args.scene, loss="none", variant=f"diag:{name}", seed=args.seed)
    rows = rows_from(metrics.evaluate(test, ref, ["psnr", "ssim", "lpips_vgg", "lpips_alex"]), key)
    rows += rows_from(convention_metrics(test, ref, metrics._net("lpips_vgg")), key)
    rows.append(dict(key, split="model", view_id=-1, metric="num_gaussians", value=float(model.num_gaussians)))
    rows.append(dict(key, split="model", view_id=-1, metric="grow_grad2d", value=float(args.grow_grad2d)))
    rows.append(dict(key, split="model", view_id=-1, metric="train_time_s", value=float(info["train_time"])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    q = df[df.split == "test"].groupby("metric").value.mean()
    log(f"DONE N={model.num_gaussians} PSNR {q['psnr']:.2f} SSIM {q['ssim']:.3f} LPIPS-A {q['lpips_alex']:.3f}")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("conventions")
    a.add_argument("config")
    b = sub.add_parser("diag")
    b.add_argument("--scene", required=True)
    b.add_argument("--grow-grad2d", type=float, default=0.0002)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--name", default=None)
    args = ap.parse_args()
    {"conventions": cmd_conventions, "diag": cmd_diag}[args.cmd](args)
