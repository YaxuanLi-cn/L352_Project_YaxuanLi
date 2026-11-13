#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
from nvsqa import datasets              
from nvsqa import gs as nvsqa_gs              
from gh import evaluate as E              
from gh import metrics as M              
from gh import train as T              
import helpers as C              
import load_finetuned as LF              
import metrics_ext as X              
X.install()
GROUPS = {"geom": ["means", "scales", "quats"], "opac": ["opacities"], "dc": ["sh0"], "sh": ["shN"]}
MULTI = ["dists", "flip", "dreamsim", "vif"]
RUNS = P.S1_RUNS
TOL = dict(psnr=0.1, ssim=0.002, ms_ssim=0.002, lpips_vgg=0.003, lpips_alex=0.003, dists=0.002, flip=0.002,
           cvvdp_image=0.03)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def eq_weight(scene_spec, loss):
    w = pd.read_csv(os.path.join(P.ORIG_TABLES, "mixed_weights.csv"))
    r = w[(w.scene == scene_spec) & (w.loss == loss)]
    return float(r.weight.iloc[0])


def build_loss(spec, scene_spec, metrics, start_vals=None, hold=("psnr", "ssim", "lpips_vgg")):
    kind, *rest = spec.split(":")
    if kind == "ctrl":
        return lambda p, g, i: (T.default_loss(p, g), {})
    if kind == "pure":
        return lambda p, g, i: (metrics.loss(rest[0], p, g), {})
    if kind == "mixed":
        loss, _, sc = rest[0].partition("@")
        w = eq_weight(scene_spec, loss) * (float(sc) if sc else 1.0)
        return lambda p, g, i: (T.default_loss(p, g) + w * metrics.loss(loss, p, g), {})
    if kind == "mad":
        target, direction = rest[0], rest[1]
        lam = float(rest[2]) if len(rest) > 2 else 1.0
        sgn = {"down": -1.0, "up": 1.0}[direction]
        def f(p, g, i):
            t = M.loss_form(target, metrics(target, p, g)).mean()
            s = start_vals[i]
            pen = sum(((metrics(k, p, g).mean() - s[k]) / TOL[k]) ** 2 for k in hold)
            scale = start_vals["_target_scale"]
            return sgn * t / scale + lam * pen, dict(target=float(t.detach()), pen=float(pen.detach()))
        return f
    if kind == "front":
        anchor, target, gamma = rest[0], rest[1], float(rest[2])
        s0 = start_vals["_anchor_scale"]
        def f(p, g, i):
            if target == "multi":
                t = sum(M.loss_form(k, metrics(k, p, g)).mean() / start_vals["_multi_scale"][k] for k in MULTI) / len(MULTI)
            else:
                t = M.loss_form(target, metrics(target, p, g)).mean()
            if anchor == "triplet":
                a = T.default_loss(p, g) / s0["default"] + metrics("lpips_vgg", p, g).mean() / s0["lpips_vgg"]
            elif anchor == "psnr":
                a = metrics("mse", p, g).mean() / s0["mse"]
            elif anchor == "default":
                a = T.default_loss(p, g) / s0["default"]
            return a - gamma * t / start_vals["_target_scale"], dict(target=float(t.detach()), anchor=float(a.detach()))
        return f
    if kind == "madl":
        target, direction = rest[0], rest[1]
        eta = float(rest[2]) if len(rest) > 2 else 0.02
        sgn = {"down": -1.0, "up": 1.0}[direction]
        lam = {k: 0.0 for k in hold}
        rho = 0.05
        def f(p, g, i):
            t = M.loss_form(target, metrics(target, p, g)).mean()
            s = start_vals[i]
            devs = {k: (metrics(k, p, g).mean() - s[k]) / TOL[k] for k in hold}
            con = sum(lam[k] * devs[k] + 0.5 * rho * devs[k] ** 2 for k in hold)
            parts = dict(target=float(t.detach()))
            for k in hold:
                dv = float(devs[k].detach())
                parts[f"dev_{k}"] = dv
                parts[f"lam_{k}"] = lam[k]
                lam[k] += eta * dv
            return sgn * t / start_vals["_target_scale"] + con, parts
        return f


def stage_b(ck, scene, loss_fn, steps, seed, groups, log_every=1000):
    torch.manual_seed(seed)
    cfg = nvsqa_gs.GSConfig(**ck["cfg"])
    model = T.load_model(ck, "cuda", requires_grad=True)
    params = model.params
    scene_scale = ck["scene_scale"]
    lrs = {"means": cfg.means_lr * scene_scale * 0.01, "scales": cfg.scales_lr, "quats": cfg.quats_lr,
           "opacities": cfg.opacities_lr, "sh0": cfg.sh0_lr, "shN": cfg.shN_lr}
    trained = {k for g in groups for k in GROUPS[g]}
    for k, p in params.items():
        p.requires_grad_(k in trained)
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items() if k in trained}
    train_ids = np.asarray(scene.train_ids)
    g = torch.Generator().manual_seed(1000 + seed)
    order, curve, status, t0, acc, parts_acc = [], [], "ok", time.time(), 0.0, {}
    for step in range(steps):
        if not order:
            order = torch.randperm(len(train_ids), generator=g).tolist()
        i = int(train_ids[order.pop()])
        pred, gt = T.render_train(model, scene, i, "cuda")
        loss, parts = loss_fn(pred, gt, i)
        if not torch.isfinite(loss):
            status = f"diverged: non-finite loss at step {step}"
            break
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None):
            status = f"diverged: non-finite gradient at step {step}"
            break
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        acc += float(loss)
        for k, v in parts.items():
            parts_acc[k] = parts_acc.get(k, 0.0) + v
        if (step + 1) % 100 == 0:
            curve.append(dict(step=step + 1, loss=acc / 100, **{k: v / 100 for k, v in parts_acc.items()}))
            acc, parts_acc = 0.0, {}
        if log_every and (step + 1) % log_every == 0:
            log(f"    step {step + 1}/{steps} {curve[-1]} {time.time() - t0:.0f}s")
    for p in params.values():
        p.requires_grad_(False)
    return model, dict(status=status, steps_done=step + 1 if status == "ok" else step, train_time=time.time() - t0, curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--loss", required=True)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="geom,opac,dc,sh")
    ap.add_argument("--init", default="base", choices=["base", "ctrl"])
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--hold", default="psnr,ssim,lpips_vgg", help="metrics held at the start value (mad losses)")
    ap.add_argument("--no-eval", action="store_true", help="render and store only (two-run protocol: no reference metric)")
    args = ap.parse_args()
    sdn = args.scene.replace("/", "_")
    groups = args.groups.split(",")
    name = args.loss.replace(":", "-").replace("@", "x") + ("" if len(groups) == 4 else "__only-" + "-".join(groups)) \
        + ("" if args.hold == "psnr,ssim,lpips_vgg" or not args.loss.startswith("mad") else "__hold-" + args.hold.replace(",", "-")) \
        + ("" if args.init == "base" else "__from-ctrl") + (f"__{args.steps}" if args.steps != 10000 else "") + f"__{args.tag}"
    d = os.path.join(RUNS, sdn, name)
    if os.path.exists(os.path.join(d, "done.json")):
        log(f"exists: {d}")
        return
    os.makedirs(d, exist_ok=True)
    scene = datasets.load_scene(args.scene)
    views = json.load(open(os.path.join(P.RERUN_VIEWS, f"{sdn}.json")))
    held = np.asarray(views["heldout"])
    held = held[np.unique(np.round(np.linspace(0, len(held) - 1, args.n_test)).astype(int))]
    tr = np.asarray(views["train"])
    tr = tr[np.unique(np.round(np.linspace(0, len(tr) - 1, args.n_train)).astype(int))]
    metrics = X.Metrics()
    if args.init == "base":
        ck = torch.load(os.path.join(P.STAGE_A_CKPT, sdn, f"base_seed{42 + args.seed}.pt"), map_location="cpu", weights_only=False)
    else:
        p = LF.load_params(os.path.join(P.RERUN_CKPT, sdn, "s0", "default__ctrl.pt"))
        ck = dict(params=p["params"], cfg=p["cfg"], scene_scale=p["scene_scale"])
    start_vals = None
    if args.loss.startswith("front:"):
        target = args.loss.split(":")[2]
        m0 = T.load_model(ck, "cuda")
        acc = {"default": [], "lpips_vgg": [], "mse": [], "t": []}
        with torch.no_grad():
            for i in np.asarray(scene.train_ids)[np.round(np.linspace(0, len(scene.train_ids) - 1, 24)).astype(int)]:
                pred, gt = T.render_train(m0, scene, int(i))
                acc["default"].append(float(T.default_loss(pred, gt)))
                acc["lpips_vgg"].append(float(metrics("lpips_vgg", pred, gt).mean()))
                acc["mse"].append(float(metrics("mse", pred, gt).mean()))
                if target == "multi":
                    for k in MULTI:
                        acc.setdefault("m_" + k, []).append(float(M.loss_form(k, metrics(k, pred, gt)).mean()))
                    acc["t"].append(1.0)
                else:
                    acc["t"].append(float(M.loss_form(target, metrics(target, pred, gt)).mean()))
        start_vals = {"_anchor_scale": {k: float(np.mean(acc[k])) for k in ("default", "lpips_vgg", "mse")}, "_target_scale": float(np.mean(acc["t"])),
                      "_multi_scale": {k: float(np.mean(acc["m_" + k])) for k in MULTI if "m_" + k in acc}}
        del m0
        torch.cuda.empty_cache()
        log(f"start scales {start_vals}")
    if args.loss.startswith("mad"):
        target = args.loss.split(":")[1]
        m0 = T.load_model(ck, "cuda")
        start_vals, tv = {}, []
        with torch.no_grad():
            sub = set(np.asarray(scene.train_ids)[np.round(np.linspace(0, len(scene.train_ids) - 1, 24)).astype(int)].tolist())
            for i in scene.train_ids:
                pred, gt = T.render_train(m0, scene, int(i))
                start_vals[int(i)] = {k: float(metrics(k, pred, gt).mean()) for k in args.hold.split(",")}
                if int(i) in sub:
                    tv.append(float(M.loss_form(target, metrics(target, pred, gt)).mean()))
        start_vals["_target_scale"] = float(np.mean(tv))
        del m0
        torch.cuda.empty_cache()
        log(f"start values of {len(tv)} training views; target loss mean {np.mean(tv):.4f}")
    fn = build_loss(args.loss, args.scene, metrics, start_vals, tuple(args.hold.split(",")))
    log(f"START {args.scene} {name}")
    model, info = stage_b(ck, scene, fn, args.steps, args.seed, groups)
    json.dump(info.pop("curve"), open(os.path.join(d, "curve.json"), "w"))
    rows = []
    for split, ids, vids in (("test", np.asarray(scene.test_ids)[held], held), ("train", tr, tr)):
        img = E.render_views(model, scene, ids)
        if split == "test":
            C.write_pngs(os.path.join(d, "test"), img, [f"{int(v):03d}.png" for v in vids], 3)
        if args.no_eval:
            continue
        ref = E.crop_ref(scene, ids)
        vals = metrics.evaluate(img, ref, M.IMAGE_METRICS + list(X.NEW_METRICS))
        for m, v in vals.items():
            rows += [dict(scene=args.scene, name=name, split=split, view=int(vw), metric=m, value=float(x)) for vw, x in zip(vids, v)]
        bias = (img.astype(np.float32) - ref.astype(np.float32)).mean(axis=(1, 2))
        for c, ch in enumerate("rgb"):
            rows += [dict(scene=args.scene, name=name, split=split, view=int(vw), metric=f"diag_bias_{ch}", value=float(x))
                     for vw, x in zip(vids, bias[:, c])]
    if rows:
        pd.DataFrame(rows).to_parquet(os.path.join(d, "metrics.parquet"))
    info.update(scene=args.scene, loss=args.loss, groups=groups, init=args.init, steps=args.steps, seed=args.seed,
                host=os.uname().nodename, heldout=[int(v) for v in held], train=[int(v) for v in tr])
    json.dump(info, open(os.path.join(d, "done.json"), "w"), indent=1)
    log(f"DONE {name} [{info['status']}] train {info['train_time']:.0f}s")
if __name__ == "__main__":
    main()
