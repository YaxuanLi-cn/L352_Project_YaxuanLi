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
import torch              
R = S.R
MULTI = ["dists", "flip", "vif", "lpips_alex", "ms_ssim"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wdefault", type=float, default=1.0, help="weight of the PSNR/SSIM term (L1 + D-SSIM) in the hold part")
    ap.add_argument("--hold-extra", default="", help="comma-separated metrics added to the hold part (round 3), e.g. flip")
    args = ap.parse_args()
    t_start = time.time()
    sdn = args.scene.replace("/", "_")
    extra = [k for k in args.hold_extra.split(",") if k]
    name = (f"ext-{args.target}-{args.gamma:g}{'' if args.wdefault == 1 else f'-w{args.wdefault:g}'}"
            f"{'' if not extra else '-h' + '+'.join(extra)}__from-ctrl__{args.steps}__s{args.seed}")
    d = os.path.join(P.S6_RUNS, sdn, name)
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
    targets = [k for k in (MULTI if args.target == "multi" else [args.target]) if k not in extra]
    names = ["lpips_vgg", "dreamsim"] + extra + targets
    m0 = R.T.load_model(ck, "cuda")
    acc = {k: [] for k in ["default"] + names}
    with torch.no_grad():
        for i in np.asarray(scene.train_ids)[np.round(np.linspace(0, len(scene.train_ids) - 1, 24)).astype(int)]:
            pred, gt = R.T.render_train(m0, scene, int(i))
            acc["default"].append(float(R.T.default_loss(pred, gt)))
            for k in names:
                acc[k].append(float(R.M.loss_form(k, metrics(k, pred, gt)).mean()))
    del m0
    torch.cuda.empty_cache()
    s0 = {k: float(np.mean(v)) for k, v in acc.items()}
    R.log(f"start values {s0}")
    def fn(pr, g, i):
        a = args.wdefault * R.T.default_loss(pr, g) / s0["default"] + R.M.loss_form("lpips_vgg", metrics("lpips_vgg", pr, g)).mean() / s0["lpips_vgg"] \
            + R.M.loss_form("dreamsim", metrics("dreamsim", pr, g)).mean() / s0["dreamsim"]                                
        for k in extra:
            a = a + R.M.loss_form(k, metrics(k, pr, g)).mean() / s0[k]
        t = sum(R.M.loss_form(k, metrics(k, pr, g)).mean() / s0[k] for k in targets) / len(targets)
        return a - args.gamma * t, dict(target=float(t.detach()), anchor=float(a.detach()))
    R.log(f"START {args.scene} {name}")
    model, info = R.stage_b(ck, scene, fn, args.steps, args.seed, ["geom", "opac", "dc", "sh"])
    json.dump(info.pop("curve"), open(os.path.join(d, "curve.json"), "w"))
    ids = np.asarray(scene.test_ids)[held]
    img = R.E.render_views(model, scene, ids)
    R.C.write_pngs(os.path.join(d, "test"), img, [f"{int(v):03d}.png" for v in held], 3)
    vals = metrics.evaluate(img, R.E.crop_ref(scene, ids), ["psnr", "ssim", "lpips_vgg", "dreamsim"])
    quick = {m: float(np.mean(v)) for m, v in vals.items()}
    json.dump(quick, open(os.path.join(d, "quick.json"), "w"))
    import aux_existing as A
    ref = R.C.read_pngs([os.path.join(A.RERUN, sdn, "reference", "test", f"{int(v):03d}.png") for v in held])
    A.table(metrics, args.scene, name, os.path.join(d, "test"), held, ref).to_parquet(os.path.join(d, "metrics.parquet"))
    info.update(scene=args.scene, anchor="triplet+dreamsim" + "".join("+" + k for k in extra), hold_extra=extra, target=args.target, targets=targets, gamma=args.gamma, wdefault=args.wdefault, init="ctrl",
                steps=args.steps, seed=args.seed, heldout=[int(v) for v in held], start_values=s0, quick=quick,
                wall_time=time.time() - t_start, finished_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(info, open(os.path.join(d, "done.json"), "w"), indent=1)
    R.log(f"DONE {name} [{info['status']}] {quick} train {info['train_time']:.0f}s wall {info['wall_time']:.0f}s")
if __name__ == "__main__":
    main()
