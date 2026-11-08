#!/usr/bin/env python
import argparse
import gc
import json
import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import helpers as C              
import numpy as np              
import pandas as pd              
import torch              
torch.backends.cudnn.benchmark = True
from nvsqa import datasets              
from gh import evaluate as E              
from gh import metrics as M              
from gh import train as T              


def conditions(cfg):
    return ([("none", "base"), ("default", "ctrl")] + [(l, "pure") for l in cfg["pure"]]
            + [(l, "mixed") for l in cfg["mixed"]])


def plan(cfg):
    jobs = []
    for spec in cfg["scenes"]:
        seeds = cfg["seed_scene_seeds"] if spec == cfg.get("seed_scene") else cfg["seeds"]
        for seed in seeds:
            for loss, variant in conditions(cfg):
                jobs.append((spec, seed, loss, variant))
    return jobs


def run_dir(cfg, spec, seed, loss, variant):
    return os.path.join(cfg["runs_root"], C.scene_dir(spec), f"s{seed}", f"{loss}__{variant}")


def stim_dir(cfg, spec, seed, loss, variant):
    return os.path.join(cfg["stimuli_root"], C.scene_dir(spec), f"s{seed}", f"{loss}__{variant}")


def original_weight(cfg, spec, seed, loss):
    d = os.path.join(cfg["orig_runs_root"], C.scene_dir(spec), f"s{seed}")
    w = json.load(open(os.path.join(d, f"{loss}__mixed", "done.json")))["mixed_weight"]
    entry = dict(weight=w)
    f = os.path.join(d, "mixed_weights.json")
    if os.path.exists(f):
        old = json.load(open(f)).get(loss)
        if old is not None:
            entry.update(default_mean=old["default_mean"], metric_loss_mean=old["metric_loss_mean"])
    return entry


def write_mixed_weights(cfg, spec, seed):
    out = os.path.join(cfg["runs_root"], C.scene_dir(spec), f"s{seed}", "mixed_weights.json")
    if os.path.exists(out):
        return
    ws = {l: original_weight(cfg, spec, seed, l) for l in cfg["mixed"]}
    ws = {l: v for l, v in ws.items() if "default_mean" in v and "metric_loss_mean" in v}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + f".{os.getpid()}"
    json.dump(ws, open(tmp, "w"), indent=1)
    os.replace(tmp, out)


def evaluate_and_store(model, scene, spec, key, metrics, sub, names, rdir, sdir, views, level):
    rows = []
    def add(split, vid, metric, value):
        rows.append(dict(key, split=split, view_id=int(vid), metric=metric, value=float(value)))
    for split, ids in (("test", scene.test_ids), ("train", sub)):
        test = E.render_views(model, scene, ids)
        ref = E.crop_ref(scene, ids)
        vals = metrics.evaluate(test, ref, names)
        for m, v in vals.items():
            for j, x in enumerate(v):
                add(split, j, m, x)
        bias = (test.astype(np.float32) - ref.astype(np.float32)).mean(axis=(1, 2))
        for j in range(len(ids)):
            for c, ch in enumerate("rgb"):
                add(split, j, f"diag_bias_{ch}", bias[j, c])
        if split == "test":
            C.write_pngs(os.path.join(sdir, "test"), test, [f"{j:03d}.png" for j in range(len(test))], level)
            if scene.fps > 0:
                rows += E._path(test, scene, spec, metrics, rdir, sdir, add)
        del test
    tr = np.asarray(views["train"])
    C.write_pngs(os.path.join(sdir, "train"), E.render_views(model, scene, tr), [f"{i:04d}.png" for i in tr], level)
    add("model", -1, "num_gaussians", model.num_gaussians)
    torch.cuda.empty_cache()
    return rows


def store_reference(cfg, scene, spec, views):
    d = os.path.join(cfg["stimuli_root"], C.scene_dir(spec), "reference")
    if os.path.exists(os.path.join(d, "done")):
        return
    C.write_pngs(os.path.join(d, "test"), E.crop_ref(scene, scene.test_ids),
                 [f"{j:03d}.png" for j in range(len(scene.test_ids))], cfg["png_compression"])
    tr = np.asarray(views["train"])
    C.write_pngs(os.path.join(d, "train"), E.crop_ref(scene, tr), [f"{i:04d}.png" for i in tr], cfg["png_compression"])
    open(os.path.join(d, "done"), "w").write("reference views of the test split and of the pilot's training views\n")


def store_ckpt(cfg, spec, seed, loss, variant, model, ck, base_path):
    if variant == "base":
        return dict(ckpt="stage-A checkpoint (unchanged)", ckpt_path=os.path.relpath(base_path, C.ROOT))
    path = os.path.join(cfg["ckpt_root"], C.scene_dir(spec), f"s{seed}", f"{loss}__{variant}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params = {k: v.detach().float().cpu() for k, v in model.params.items()}
    if spec in cfg["ckpt_fp16_delta"]:
        enc = "fp16_delta_from_base"
        params = {k: (v - ck["params"][k].float()).half() for k, v in params.items()}
    else:
        enc = "fp32"
    obj = dict(params=params, encoding=enc, base=os.path.relpath(base_path, C.ROOT), scene=spec, seed=seed, loss=loss,
               variant=variant, cfg=ck["cfg"], scene_scale=ck["scene_scale"],
               note="fp16_delta_from_base: parameters = base parameters + delta (float32)" if enc != "fp32" else "")
    torch.save(obj, path)
    return dict(ckpt=enc, ckpt_path=os.path.relpath(path, C.ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="walk through the scenes in reverse order")
    ap.add_argument("--only", default="", help="substring filter on '<scene> s<seed> <loss>/<variant>'")
    ap.add_argument("--steps", type=int, default=None, help="override stage_b_steps (tests only)")
    ap.add_argument("--first", default="", help="'|'-separated substrings of '<scene> s<seed>': runs that match an "
                                                "earlier one are taken first (several workers, several GPUs)")
    ap.add_argument("--skip", default="", help="'|'-separated substrings of '<scene> s<seed>': runs this worker never takes")
    ap.add_argument("--first-conds", default="", help="'|'-separated '<loss>/<variant>' of seed 0 that are taken before "
                                                      "everything else (the models needed first)")
    args = ap.parse_args()
    cfg = C.load_cfg(args.config)
    if args.steps is not None:
        cfg["stage_b_steps"] = args.steps
    jobs = [j for j in plan(cfg) if args.only in f"{j[0]} s{j[1]} {j[2]}/{j[3]}"]
    if args.reverse:
        order = list(dict.fromkeys(j[0] for j in jobs))[::-1]
        jobs = [j for s in order for j in jobs if j[0] == s]
    if args.skip:
        skip = [p for p in args.skip.split("|") if p]
        jobs = [j for j in jobs if not any(p in f"{j[0]} s{j[1]}" for p in skip)]
    if args.first:
        pri = [p for p in args.first.split("|") if p]
        def rank(j):
            key = f"{j[0]} s{j[1]}"
            return next((i for i, p in enumerate(pri) if p in key), len(pri))
        jobs = sorted(jobs, key=rank)
    if args.first_conds:
        fc = set(args.first_conds.split("|"))
        jobs = sorted(jobs, key=lambda j: 0 if (j[1] == 0 and f"{j[2]}/{j[3]}" in fc) else (1 if j[1] == 0 else 2))
    if args.list:
        for j in jobs:
            d = run_dir(cfg, *j)
            print("done" if os.path.exists(os.path.join(d, "done.json")) else "pending", *j)
        return
    torch.set_num_threads(4)
    metrics = M.Metrics()
    lock_root = os.path.join(cfg["runs_root"], ".locks")
    os.makedirs(lock_root, exist_ok=True)
    cache = {}
    def get_scene(spec):
        if spec not in cache:
            cache.clear()
            gc.collect()
            cache[spec] = datasets.load_scene(spec)
            C.log(f"loaded {spec}: {tuple(cache[spec].images.shape)}")
        return cache[spec]
    def finished(j):
        d = run_dir(cfg, *j)
        return os.path.exists(os.path.join(d, "done.json"))
    while True:
        left = [j for j in jobs if not finished(j)]
        if not left:
            break
        progressed = False
        for spec, seed, loss, variant in left:
            if finished((spec, seed, loss, variant)):
                continue
            d = run_dir(cfg, spec, seed, loss, variant)
            lock = os.path.join(lock_root, f"{C.scene_dir(spec)}__s{seed}__{loss}__{variant}")
            if not C.acquire(lock):
                continue
            progressed = True
            base_path = os.path.join(cfg["base_ckpt_root"], C.scene_dir(spec), f"base_seed{42 + seed}.pt")
            ck = torch.load(base_path, map_location="cpu", weights_only=False)
            scene = get_scene(spec)
            views = C.views_for(scene, spec, cfg)
            store_reference(cfg, scene, spec, views)
            write_mixed_weights(cfg, spec, seed)
            os.makedirs(d, exist_ok=True)
            key = dict(scene=spec, loss=loss, variant=variant, seed=seed)
            sub = datasets.subsample_views(scene.train_ids, cfg["train_subset"])
            info = dict(key, base_num_gaussians=ck["num_gaussians"], stage_b_steps=cfg["stage_b_steps"],
                        base_ckpt=os.path.relpath(base_path, C.ROOT))
            C.log(f"START {spec} s{seed} {loss}/{variant}")
            t0 = time.time()
            if variant == "base":
                model = T.load_model(ck)
                tinfo = dict(status="ok", steps_done=0, train_time=0.0, curve=[])
            else:
                w = None
                if variant == "mixed":
                    w = original_weight(cfg, spec, seed, loss)["weight"]
                    info["mixed_weight"] = w
                fn = T.make_loss(loss, variant, metrics, w)
                model, tinfo = T.stage_b(ck, scene, fn, cfg["stage_b_steps"], seed, log=C.log,
                                         log_every=cfg["log_every"])
            info.update({k: v for k, v in tinfo.items() if k != "curve"})
            json.dump(tinfo.get("curve", []), open(os.path.join(d, "curve.json"), "w"))
            info.update(store_ckpt(cfg, spec, seed, loss, variant, model, ck, base_path))
            te = time.time()
            rows = evaluate_and_store(model, scene, spec, key, metrics, sub, cfg["image_metrics"], d,
                                      stim_dir(cfg, spec, seed, loss, variant), views, cfg["png_compression"])
            rows.append(dict(key, split="model", view_id=-1, metric="steps_done", value=tinfo["steps_done"]))
            rows.append(dict(key, split="model", view_id=-1, metric="train_time_s", value=tinfo["train_time"]))
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(d, "metrics.csv"), index=False)
            info.update(eval_time=time.time() - te, total_time=time.time() - t0)
            json.dump(info, open(os.path.join(d, "done.json"), "w"), indent=1)
            q = df[df.split == "test"].groupby("metric").value.mean()
            C.log(f"DONE  {spec} s{seed} {loss}/{variant} [{tinfo['status']}] train {tinfo['train_time']:.0f}s "
                  f"eval {info['eval_time']:.0f}s  PSNR {q.get('psnr', np.nan):.2f} LPIPS-A {q.get('lpips_alex', np.nan):.4f}")
            del model, ck
            torch.cuda.empty_cache()
            C.release(lock)
        if not progressed:
            time.sleep(60)
    C.log("RERUN WORKER FINISHED")
if __name__ == "__main__":
    main()
