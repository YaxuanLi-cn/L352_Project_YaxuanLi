#!/usr/bin/env python
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
sys.path.insert(0, HERE)
import aux_existing as A              
import numpy as np              
import pandas as pd              
import torch              
import torch.nn.functional as F              
S, R = A.S, A.R
SCENE = sys.argv[1] if len(sys.argv) > 1 else "gsqa/truck"
EPS, STEPS = 2.0 / 255.0, 20


def blur(x, sigma=1.5):
    k = int(2 * round(3 * sigma) + 1)
    g = torch.exp(-(torch.arange(k, device=x.device) - k // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).float()
    x = F.pad(x, (k // 2,) * 4, mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(x, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def tint(x, seed, cell=64, amp=6.0 / 255.0):
    h, w = x.shape[-2:]
    g = torch.Generator().manual_seed(int(seed))
    f = torch.randn(1, 3, h // cell + 2, w // cell + 2, generator=g).to(x.device)
    f = F.interpolate(f, size=(h, w), mode="bicubic", align_corners=False)
    return (x + amp * f / f.abs().amax()).clamp(0, 1)


def adversarial(metrics, name, x, r):
    adv = x.clone()
    for _ in range(STEPS):
        adv.requires_grad_(True)
        (g,) = torch.autograd.grad(R.M.loss_form(name, metrics(name, adv, r)).mean(), adv)
        adv = (x + (adv.detach() + EPS / 4 * g.sign() - x).clamp(-EPS, EPS)).clamp(0, 1)
    return adv


def main():
    sdn = SCENE.replace("/", "_")
    out = os.path.join(P.S2_AUX, f"controls_{sdn}.parquet")
    png = os.path.join(P.S2_AUX, "controls_png" if sdn == "gsqa_truck" else f"controls_png_{sdn}")
    if os.path.exists(out):
        return
    held = A.held_views(sdn)
    metrics = R.X.Metrics()
    ref = R.C.read_pngs([os.path.join(A.RERUN, sdn, "reference", "test", f"{int(v):03d}.png") for v in held])
    img = R.C.read_pngs([os.path.join(A.RERUN, sdn, "s0", "default__ctrl", "test", f"{int(v):03d}.png") for v in held])
    made = {k: [] for k in ("blur15", "tint", "adv_dists", "adv_dreamsim", "adv_flip")}
    for i, v in enumerate(held):
        x = torch.as_tensor(img[i:i + 1]).cuda().permute(0, 3, 1, 2).float() / 255.0
        r = torch.as_tensor(ref[i:i + 1]).cuda().permute(0, 3, 1, 2).float() / 255.0
        cond = {"blur15": blur(x), "tint": tint(x, v)}
        for name in ("dists", "dreamsim", "flip"):
            cond["adv_" + name] = adversarial(metrics, name, x, r)
        for k, y in cond.items():
            made[k].append((y.detach().clamp(0, 1) * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()[0])
    rows = []
    for k, stack in made.items():
        stack = np.stack(stack)
        R.C.write_pngs(os.path.join(png, k), stack, [f"{int(v):03d}.png" for v in held], 3)
        for (res, m), val in S.evaluate_both(metrics, stack, ref).items():
            rows += [dict(scene=SCENE, name=k, split="test", view=int(vw), res=res, metric=m, value=float(x)) for vw, x in zip(held, val)]
        R.log(f"{k}: max |change| {np.abs(stack.astype(int) - img.astype(int)).max()} levels")
    pd.DataFrame(rows).to_parquet(out)
if __name__ == "__main__":
    main()
