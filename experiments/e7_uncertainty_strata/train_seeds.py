#!/usr/bin/env python
import argparse
import os
import sys
import time
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
ROOT = P.ROOT
import yaml              


def log(msg):
    print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def plan(cfg):
    out = []
    for spec in cfg["scenes"]:
        d = spec.replace("/", "_")
        have = []
        for s in (cfg.get("extra_seeds_by_scene") or {}).get(spec, cfg["extra_seeds"]):
            p_old = os.path.join(ROOT, cfg["existing_seed_ckpts"], d, f"base_seed{s}.pt")
            p_new = os.path.join(ROOT, cfg["new_seed_ckpts"], d, f"base_seed{s}.pt")
            if os.path.exists(p_old):
                have.append((spec, s, p_old, True))
            else:
                have.append((spec, s, p_new, os.path.exists(p_new)))
        out += have[:cfg["seed_target"]]
    return out


def acquire(lock):
    try:
        os.makedirs(lock)
        return True
    except FileExistsError:
        return False


def release(lock):
    os.rmdir(lock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="walk through the jobs in reverse order (second worker)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    jobs = plan(cfg)
    if args.reverse:
        jobs = jobs[::-1]
    if args.list:
        for spec, s, p, ok in jobs:
            print("exists " if ok else "missing", spec, s, os.path.relpath(p, ROOT))
        return
    import torch
    from nvsqa import datasets
    from gh import train as T
    lock_root = os.path.join(ROOT, cfg["new_seed_ckpts"], ".locks")
    os.makedirs(lock_root, exist_ok=True)
    scene = None
    for spec, s, path, ok in jobs:
        if ok or os.path.exists(path):
            continue
        lock = os.path.join(lock_root, f"{spec.replace('/', '_')}_{s}")
        if not acquire(lock):
            continue
        if scene is None or scene.name != spec.split("/")[1]:
            scene = datasets.load_scene(spec)
        log(f"START {spec} seed {s}")
        ck = T.stage_a(scene, s, path, steps=cfg["stage_a_steps"], log=log)
        log(f"DONE  {spec} seed {s}: N={ck['num_gaussians']} {ck['train_time'] / 60:.1f} min")
        del ck
        torch.cuda.empty_cache()
        release(lock)
    log("SEED TRAINING FINISHED")
if __name__ == "__main__":
    main()
