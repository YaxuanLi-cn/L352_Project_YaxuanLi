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
sys.path.insert(0, P.code(P.E5, P.S2))                                                      
import run_suff as S              
import numpy as np              
import pandas as pd              
R = S.R
T_TRUCK = 0.0191150705019633                                     
GAMMA_TRUCK = 0.0125
METRICS = ["psnr", "ssim", "lpips_vgg", "dreamsim"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--mult", type=float, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()
    sdn = args.scene.replace("/", "_")
    name = f"triplet-dreamsim-aligned-x{args.mult:g}__from-ctrl__{args.steps}__s{args.seed}"
    d = os.path.join(P.S4_RUNS, sdn, name)
    if os.path.exists(os.path.join(d, "done.json")):
        R.log(f"exists: {d}")
        return
    os.makedirs(d, exist_ok=True)
    scene = R.datasets.load_scene(args.scene)
    views = json.load(open(os.path.join(P.RERUN_VIEWS, f"{sdn}.json")))
    held = np.asarray(views["heldout"])
    held = held[np.unique(np.round(np.linspace(0, len(held) - 1, 20)).astype(int))]
    metrics = R.X.Metrics()
    p = R.LF.load_params(os.path.join(P.RERUN_CKPT, sdn, "s0", "default__ctrl.pt"))
    ck = dict(params=p["params"], cfg=p["cfg"], scene_scale=p["scene_scale"])
    start_vals = S.start_values(argparse.Namespace(loss="front:triplet:dreamsim:0"), ck, scene, metrics)
    t0 = start_vals["_target_scale"]
    gamma = float(f"{args.mult * GAMMA_TRUCK * t0 / T_TRUCK:.4g}")
    loss = f"front:triplet:dreamsim:{gamma!r}"
    R.log(f"start scales {start_vals}; DreamSim start value {t0:.6f} -> gamma0 = {GAMMA_TRUCK * t0 / T_TRUCK:.6g}, this run gamma = {gamma!r}")
    fn = R.build_loss(loss, args.scene, metrics, start_vals)
    R.log(f"START {args.scene} {name}")
    model, info = R.stage_b(ck, scene, fn, args.steps, args.seed, ["geom", "opac", "dc", "sh"])
    json.dump(info.pop("curve"), open(os.path.join(d, "curve.json"), "w"))
    ids = np.asarray(scene.test_ids)[held]
    img = R.E.render_views(model, scene, ids)
    R.C.write_pngs(os.path.join(d, "test"), img, [f"{int(v):03d}.png" for v in held], 3)
    vals = metrics.evaluate(img, R.E.crop_ref(scene, ids), METRICS)
    rows = [dict(scene=args.scene, name=name, split="test", view=int(vw), res="full", metric=m, value=float(x)) for m, v in vals.items() for vw, x in zip(held, v)]
    pd.DataFrame(rows).to_parquet(os.path.join(d, "metrics.parquet"))
    info.update(scene=args.scene, loss=loss, anchor="triplet", target="dreamsim", mult=args.mult, gamma=gamma, gamma0=GAMMA_TRUCK * t0 / T_TRUCK,
                dreamsim_start_train=t0, dreamsim_start_truck=T_TRUCK, init="ctrl", steps=args.steps, seed=args.seed, heldout=[int(v) for v in held],
                start_scales=start_vals, wall_time=time.time() - t_start, finished_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(info, open(os.path.join(d, "done.json"), "w"), indent=1)
    R.log(f"DONE {name} [{info['status']}] gamma {gamma!r} train {info['train_time']:.0f}s wall {info['wall_time']:.0f}s")
if __name__ == "__main__":
    main()
