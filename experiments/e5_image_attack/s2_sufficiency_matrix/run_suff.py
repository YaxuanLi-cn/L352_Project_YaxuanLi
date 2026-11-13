#!/usr/bin/env python
import argparse
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
ROOT = P.ROOT
sys.path.insert(0, P.code(P.E5, P.S1))
import run_ft as R              
import numpy as np              
import pandas as pd              
import torch              
import torch.nn.functional as F              
RUNS = P.S2_RUNS
IMG = list(R.M.IMAGE_METRICS)
@torch.no_grad()


def evaluate_both(metrics, img_u8, ref_u8, names=IMG):
    out = {(res, n): [] for res in ("full", "half") for n in names}
    for i in range(len(img_u8)):
        t = torch.as_tensor(np.asarray(img_u8[i:i + 1])).cuda().permute(0, 3, 1, 2).float() / 255.0
        r = torch.as_tensor(np.asarray(ref_u8[i:i + 1])).cuda().permute(0, 3, 1, 2).float() / 255.0
        for res, (a, b) in (("full", (t, r)), ("half", (F.avg_pool2d(t, 2), F.avg_pool2d(r, 2)))):
            for n in names:
                out[(res, n)].append(float(metrics(n, a.double(), b.double())[0]) if n == "vif" else float(metrics(n, a, b)[0]))
    return out


def start_values(args, ck, scene, metrics):
    target = args.loss.split(":")[2]
    m0 = R.T.load_model(ck, "cuda")
    acc = {"default": [], "lpips_vgg": [], "mse": [], "t": []}
    with torch.no_grad():
        for i in np.asarray(scene.train_ids)[np.round(np.linspace(0, len(scene.train_ids) - 1, 24)).astype(int)]:
            pred, gt = R.T.render_train(m0, scene, int(i))
            acc["default"].append(float(R.T.default_loss(pred, gt)))
            acc["lpips_vgg"].append(float(metrics("lpips_vgg", pred, gt).mean()))
            acc["mse"].append(float(metrics("mse", pred, gt).mean()))
            acc["t"].append(float(R.M.loss_form(target, metrics(target, pred, gt)).mean()))
    del m0
    torch.cuda.empty_cache()
    return {"_anchor_scale": {k: float(np.mean(acc[k])) for k in ("default", "lpips_vgg", "mse")}, "_target_scale": float(np.mean(acc["t"])),
            "_multi_scale": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--loss", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=10)
    args = ap.parse_args()
    kind, anchor, target, gamma = args.loss.split(":")
    t_start = time.time()
    sdn = args.scene.replace("/", "_")
    name = args.loss.replace(":", "-") + f"__from-ctrl__{args.steps}__s{args.seed}"
    d = os.path.join(RUNS, sdn, name)
    if os.path.exists(os.path.join(d, "done.json")):
        R.log(f"exists: {d}")
        return
    os.makedirs(d, exist_ok=True)
    scene = R.datasets.load_scene(args.scene)
    views = json.load(open(os.path.join(P.RERUN_VIEWS, f"{sdn}.json")))
    held = np.asarray(views["heldout"])
    held = held[np.unique(np.round(np.linspace(0, len(held) - 1, args.n_test)).astype(int))]
    tr = np.asarray(views["train"])
    tr = tr[np.unique(np.round(np.linspace(0, len(tr) - 1, args.n_train)).astype(int))]
    metrics = R.X.Metrics()
    p = R.LF.load_params(os.path.join(P.RERUN_CKPT, sdn, "s0", "default__ctrl.pt"))
    ck = dict(params=p["params"], cfg=p["cfg"], scene_scale=p["scene_scale"])
    start_vals = start_values(args, ck, scene, metrics)
    R.log(f"start scales {start_vals}")
    fn = R.build_loss(args.loss, args.scene, metrics, start_vals)
    R.log(f"START {args.scene} {name}")
    model, info = R.stage_b(ck, scene, fn, args.steps, args.seed, ["geom", "opac", "dc", "sh"])
    json.dump(info.pop("curve"), open(os.path.join(d, "curve.json"), "w"))
    t_eval = time.time()
    rows = []
    for split, ids, vids in (("test", np.asarray(scene.test_ids)[held], held), ("train", tr, tr)):
        img = R.E.render_views(model, scene, ids)
        if split == "test":
            R.C.write_pngs(os.path.join(d, "test"), img, [f"{int(v):03d}.png" for v in vids], 3)
        ref = R.E.crop_ref(scene, ids)
        for (res, m), v in evaluate_both(metrics, img, ref).items():
            rows += [dict(scene=args.scene, name=name, split=split, view=int(vw), res=res, metric=m, value=float(x)) for vw, x in zip(vids, v)]
    pd.DataFrame(rows).to_parquet(os.path.join(d, "metrics.parquet"))
    info.update(scene=args.scene, loss=args.loss, anchor=anchor, target=target, gamma=float(gamma), init="ctrl", steps=args.steps, seed=args.seed,
                host=os.uname().nodename, heldout=[int(v) for v in held], train=[int(v) for v in tr], start_scales=start_vals,
                eval_time=time.time() - t_eval, wall_time=time.time() - t_start, finished_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(info, open(os.path.join(d, "done.json"), "w"), indent=1)
    R.log(f"DONE {name} [{info['status']}] train {info['train_time']:.0f}s eval {info['eval_time']:.0f}s wall {info['wall_time']:.0f}s")
if __name__ == "__main__":
    main()
