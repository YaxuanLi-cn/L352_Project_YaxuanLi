import glob
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
sys.path.insert(0, os.path.join(HERE, ".."))
import helpers as C              
import drift as D              
import numpy as np              
from nvsqa import video_io              
SCENES = ["gsqa/truck", "gsqa/train", "gsqa/playground2", "gsqa/flowers",
          "mip360/kitchen", "mip360/bonsai", "mip360/garden", "mip360/bicycle"]
CTRL = "default__ctrl"
OUT = os.path.join(P.DECOMPOSITION, "out")
RES = os.path.join(P.DECOMPOSITION, "results")


def models():
    return [CTRL] + D.loss_models()


def held_out(spec):
    return D.held_out(spec)


def pick(views, n):
    if n is None or len(views) <= n:
        return list(views)
    idx = np.unique(np.round(np.linspace(0, len(views) - 1, n)).astype(int))
    return [views[i] for i in idx]


def reference(spec, views):
    sd = C.scene_dir(spec)
    if spec.startswith("gsqa/"):
        ref = video_io.read_video(os.path.join(P.PART1_RENDERS, spec, "reference.mp4"))[0]
        return np.ascontiguousarray(ref[views])
    rf = sorted(glob.glob(os.path.join(D.RERUN, sd, "reference", "test", "*.png")))
    return C.read_pngs([rf[v] for v in views])


def pair(spec, model, views):
    A = D.frames("paper", spec, model, views)
    B = D.frames("rerun", spec, model, views)
    return A, B
