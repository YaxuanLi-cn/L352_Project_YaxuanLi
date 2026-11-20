#!/usr/bin/env python
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import helpers as C              
import numpy as np              
import torch              
from nvsqa import datasets              
from gh import evaluate as E              
from gh import train as T              
import train_seeds              


def main():
    rcfg = C.load_cfg(sys.argv[2])
    import yaml
    jobs = train_seeds.plan(yaml.safe_load(open(sys.argv[1])))
    if len(sys.argv) > 3:
        jobs = [j for j in jobs if j[0] in sys.argv[3:]]
    scene = None
    for spec, seed, path, _ in jobs:
        out = os.path.join(rcfg["stimuli_root"], C.scene_dir(spec), "seeds", str(seed))
        if os.path.exists(os.path.join(out, "done")):
            continue
        if not os.path.exists(path):
            C.log(f"missing checkpoint {spec} seed {seed}: {path}")
            continue
        if scene is None or f"{scene.dataset}/{scene.name}" != spec:
            scene = datasets.load_scene(spec)
        views = C.views_for(scene, spec, rcfg)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = T.load_model(ck)
        C.write_pngs(os.path.join(out, "test"), E.render_views(model, scene, scene.test_ids),
                     [f"{j:03d}.png" for j in range(len(scene.test_ids))], rcfg["png_compression"])
        tr = np.asarray(views["train"])
        C.write_pngs(os.path.join(out, "train"), E.render_views(model, scene, tr), [f"{i:04d}.png" for i in tr],
                     rcfg["png_compression"])
        open(os.path.join(out, "done"), "w").write(f"{os.path.relpath(path, C.ROOT)}\nN={ck['num_gaussians']}\n")
        C.log(f"rendered {spec} seed {seed} ({len(scene.test_ids)} test + {len(tr)} training views)")
        del model, ck
        torch.cuda.empty_cache()
    C.log("SEED RENDERING FINISHED")
if __name__ == "__main__":
    main()
