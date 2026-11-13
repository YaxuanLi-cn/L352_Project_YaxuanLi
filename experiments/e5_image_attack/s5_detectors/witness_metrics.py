#!/usr/bin/env python
import glob
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
sys.path.insert(0, P.code(P.E5, P.S2))                                                
import aux_existing as A              
R = A.R
SCENES = ["gsqa/truck", "gsqa/flowers", "gsqa/playground2", "gsqa/train"]


def main():
    metrics = R.X.Metrics()
    for scene in SCENES:
        sdn = scene.replace("/", "_")
        held = A.held_views(sdn)
        ref = R.C.read_pngs([os.path.join(A.RERUN, sdn, "reference", "test", f"{int(v):03d}.png") for v in held])
        todo = [("control", os.path.join(A.RERUN, sdn, "s0", "default__ctrl", "test")),
                ("lpips_vgg__mixed", os.path.join(A.RERUN, sdn, "s0", "lpips_vgg__mixed", "test"))]
        for d in sorted(glob.glob(os.path.join(P.S4_RUNS, sdn, "*"))):
            if os.path.exists(os.path.join(d, "done.json")):
                todo.append((os.path.basename(d), os.path.join(d, "test")))
        for d in sorted(glob.glob(os.path.join(P.S2_RUNS, sdn, "front-triplet-*__s0"))):
            if os.path.exists(os.path.join(d, "done.json")):
                todo.append((os.path.basename(d), os.path.join(d, "test")))
        for name, d in todo:
            out = os.path.join(P.S5_REC, "witness", f"{sdn}__{name}.parquet")
            if os.path.exists(out):
                continue
            t = A.table(metrics, scene, name, d, held, ref)
            t.to_parquet(out)
            f = t[t.res == "full"].groupby("metric").value.mean()
            R.log(f"{scene} {name}: psnr {f['psnr']:.3f} ssim {f['ssim']:.4f} lpips_vgg {f['lpips_vgg']:.4f} dreamsim {f['dreamsim']:.4f} flip {f['flip']:.4f} cvvdp {f['cvvdp_image']:.3f}")
if __name__ == "__main__":
    main()
