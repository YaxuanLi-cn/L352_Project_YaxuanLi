#!/usr/bin/env python
import argparse
import gc
import json
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
import numpy as np              
import pandas as pd              
import torch              
torch.backends.cudnn.benchmark = True
import yaml              
from nvsqa import datasets              
from gh import evaluate as E              
from gh import metrics as M              
from gh import train as T              


def log(msg):
    print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def acquire(lock):
    try:
        os.makedirs(lock)
        return True
    except FileExistsError:
        return False


def release(lock):
    os.rmdir(lock)


def load_cfg(path):
    cfg = yaml.safe_load(open(path))
    for k in ("runs_root", "ckpt_root", "stimuli_root", "results_dir"):
        cfg[k] = os.path.normpath(os.path.join(ROOT, cfg[k]))
    return cfg


def conditions(cfg):
    out = [("none", "base")] if cfg.get("include_base", True) else []
    out.append(("default", "ctrl"))
    out += [(l, "pure") for l in cfg.get("pure") or []]
    out += [(l, "mixed") for l in cfg.get("mixed") or []]
    for l, scales in (cfg.get("mixed_scales") or {}).items():
        out += [(l, f"mixed@{sc:g}") for sc in scales]
    return out


def base_weight(cfg, spec, seed, loss, compute):
    roots = [cfg["runs_root"]] + ([os.path.normpath(os.path.join(ROOT, cfg["weights_from"]))] if cfg.get("weights_from") else [])
    for root in roots:
        sdir_ = os.path.join(root, spec.replace("/", "_"), f"s{seed}")
        wfile = os.path.join(sdir_, f"mixed_weight_{loss}.json")
        legacy = os.path.join(sdir_, "mixed_weights.json")
        if os.path.exists(wfile):
            return json.load(open(wfile))
        if os.path.exists(legacy) and loss in json.load(open(legacy)):
            return json.load(open(legacy))[loss]
    entry = compute()
    wfile = os.path.join(cfg["runs_root"], spec.replace("/", "_"), f"s{seed}", f"mixed_weight_{loss}.json")
    os.makedirs(os.path.dirname(wfile), exist_ok=True)
    with open(wfile + f".{os.getpid()}", "w") as fw:
        json.dump(entry, fw, indent=1)
    os.replace(wfile + f".{os.getpid()}", wfile)
    return entry


def plan(cfg):
    jobs = []
    for spec in cfg["scenes"]:
        seeds = cfg["seed_scene_seeds"] if spec == cfg.get("seed_scene") else cfg["seeds"]
        for seed in seeds:
            for loss, variant in conditions(cfg):
                jobs.append((spec, seed, loss, variant))
    return jobs


def run_dir(cfg, spec, seed, loss, variant):
    return os.path.join(cfg["runs_root"], spec.replace("/", "_"), f"s{seed}", f"{loss}__{variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only-stage-a", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    jobs = plan(cfg)
    if args.list:
        for j in jobs:
            d = run_dir(cfg, *j)
            st = "done" if os.path.exists(os.path.join(d, "done.json")) else \
                "running" if os.path.exists(d + ".lock") else "pending"
            print(st, *j)
        return
    torch.set_num_threads(4)
    metrics = M.Metrics()
    lock_root = os.path.join(cfg["runs_root"], ".locks")
    os.makedirs(lock_root, exist_ok=True)
    scene_cache = {}
    def get_scene(spec):
        if spec not in scene_cache:
            scene_cache.clear()
            gc.collect()
            scene_cache[spec] = datasets.load_scene(spec)
            s = scene_cache[spec]
            log(f"loaded {spec}: {tuple(s.images.shape)}, {len(s.train_ids)} train / {len(s.test_ids)} test")
        return scene_cache[spec]
    def get_base(spec, seed):
        path = os.path.join(cfg["ckpt_root"], spec.replace("/", "_"), f"base_seed{42 + seed}.pt")
        if not os.path.exists(path):
            lock = os.path.join(cfg["ckpt_root"], ".locks", f"{spec.replace('/', '_')}_{seed}")
            os.makedirs(os.path.dirname(lock), exist_ok=True)
            if not acquire(lock):
                return None
            if not os.path.exists(path):
                log(f"STAGE A {spec} seed {42 + seed}")
                ck = T.stage_a(get_scene(spec), 42 + seed, path, steps=cfg["stage_a_steps"], log=log,
                               half=cfg.get("ckpt_half", False))
                log(f"STAGE A done {spec}: N={ck['num_gaussians']} {ck['train_time'] / 60:.1f} min")
            release(lock)
        return torch.load(path, map_location="cpu", weights_only=False)
    if args.only_stage_a:
        for spec in cfg["scenes"]:
            seeds = cfg["seed_scene_seeds"] if spec == cfg.get("seed_scene") else cfg["seeds"]
            for seed in seeds:
                get_base(spec, seed)
        return
    base_cache = {}
    while True:
        left = [jb for jb in jobs if not os.path.exists(os.path.join(run_dir(cfg, *jb), "done.json"))]
        if not left:
            break
        n_before = len(left)
        for spec, seed, loss, variant in jobs:
            d = run_dir(cfg, spec, seed, loss, variant)
            if os.path.exists(os.path.join(d, "done.json")):
                continue
            lock = os.path.join(lock_root, os.path.relpath(d, cfg["runs_root"]).replace("/", "__"))
            if not acquire(lock):
                continue
            if (spec, seed) not in base_cache:
                base_cache.clear()
                ck = get_base(spec, seed)
                if ck is None:
                    release(lock)
                    continue
                base_cache[(spec, seed)] = ck
            ck = base_cache[(spec, seed)]
            scene = get_scene(spec)
            os.makedirs(d, exist_ok=True)
            key = dict(scene=spec, loss=loss, variant=variant, seed=seed)
            sub = datasets.subsample_views(scene.train_ids, cfg["train_subset"])
            info = dict(key, base_num_gaussians=ck["num_gaussians"], stage_b_steps=cfg["stage_b_steps"])
            log(f"START {spec} s{seed} {loss}/{variant}")
            t0 = time.time()
            if variant == "base":
                model = T.load_model(ck)
                tinfo = dict(status="ok", steps_done=0, train_time=0.0, curve=[])
            else:
                w = None
                if variant.startswith("mixed"):
                    def compute():
                        w0, dmean, mmean = T.mixed_weight(ck, scene, sub, loss, metrics)
                        return dict(weight=w0, default_mean=dmean, metric_loss_mean=mmean,
                                    views=[int(i) for i in sub])
                    entry = base_weight(cfg, spec, seed, loss, compute)
                    scale = float(variant.split("@")[1]) if "@" in variant else 1.0
                    w = entry["weight"] * scale
                    info.update(mixed_weight=w, mixed_weight_base=entry["weight"], mixed_scale=scale)
                fn = T.make_loss(loss, variant, metrics, w)
                model, tinfo = T.stage_b(ck, scene, fn, cfg["stage_b_steps"], seed, log=log,
                                         log_every=cfg.get("log_every", 1000))
            info.update({k: v for k, v in tinfo.items() if k != "curve"})
            json.dump(tinfo.get("curve", []), open(os.path.join(d, "curve.json"), "w"))
            stim = None
            if cfg.get("stimuli") and seed in cfg.get("stimuli_seeds", [0]):
                stim = os.path.join(cfg["stimuli_root"], spec.replace("/", "_"), f"{loss}__{variant}")
            te = time.time()
            rows = E.evaluate(model, scene, spec, key, metrics, sub, cfg["image_metrics"], d, stim)
            rows.append(dict(key, split="model", view_id=-1, metric="steps_done", value=tinfo["steps_done"]))
            rows.append(dict(key, split="model", view_id=-1, metric="train_time_s", value=tinfo["train_time"]))
            pd.DataFrame(rows).to_csv(os.path.join(d, "metrics.csv"), index=False)
            info.update(eval_time=time.time() - te, total_time=time.time() - t0)
            json.dump(info, open(os.path.join(d, "done.json"), "w"), indent=1)
            df = pd.DataFrame(rows)
            q = df[df.split == "test"].groupby("metric").value.mean()
            log(f"DONE  {spec} s{seed} {loss}/{variant} [{tinfo['status']}] train {tinfo['train_time']:.0f}s "
                f"eval {info['eval_time']:.0f}s  PSNR {q.get('psnr', np.nan):.2f} LPIPS-A {q.get('lpips_alex', np.nan):.4f}")
            del model
            torch.cuda.empty_cache()
            release(lock)
        left = [jb for jb in jobs if not os.path.exists(os.path.join(run_dir(cfg, *jb), "done.json"))]
        if left and len(left) == n_before:
            time.sleep(60)
    log("WORKER FINISHED")
if __name__ == "__main__":
    main()
