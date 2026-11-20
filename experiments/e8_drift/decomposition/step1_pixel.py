#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib as L              
OUT = L.OUT
TILE = 16
DEV = "cpu"
torch.set_num_threads(2)


def t(x):
    return torch.from_numpy(x).to(DEV).float() / 255.0


def tiles(m):
    n, h, w = m.shape
    h2, w2 = h // TILE * TILE, w // TILE * TILE
    return m[:, :h2, :w2].reshape(n, h2 // TILE, TILE, w2 // TILE, TILE).mean(dim=(2, 4)).reshape(n, -1)


def corr(a, b):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-20)).cpu().numpy()


def main():
    spec = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    sd = spec.replace("/", "_")
    views = L.held_out(spec)
    G = t(L.reference(spec, views))
    rows, trow = [], []
    ctrl = None
    for model in L.models():
        A8, B8 = L.pair(spec, model, views)
        A, B = t(A8), t(B8)
        Mn = 0.5 * (A + B)
        f = lambda x, y: ((x - y) ** 2).mean(dim=(1, 2, 3)).double().cpu().numpy()              
        ag, bg, ab, mg = f(A, G), f(B, G), f(A, B), f(Mn, G)
        for i, v in enumerate(views):
            rows.append(dict(scene=spec, model=model, view=int(v), mse_AG=ag[i], mse_BG=bg[i], mse_AB=ab[i], mse_MG=mg[i]))
        eA, eB, eM = ((x - G).abs().mean(dim=3) for x in (A, B, Mn))
        if model == L.CTRL:
            ctrl = (eA, eB, eM)
        else:
            d = tiles((A - B).abs().mean(dim=3))
            exA, exB, exM = tiles(eA - ctrl[0]), tiles(eB - ctrl[1]), tiles(eM - ctrl[2])
            rawA, rawB = tiles(eA), tiles(eB)
            p_ex = 0.5 * (corr(d, exA) + corr(d, exB))
            p_exM = corr(d, exM)
            p_raw = 0.5 * (corr(d, rawA) + corr(d, rawB))
            dn, exn, exMn = d.cpu().numpy(), 0.5 * (exA + exB).cpu().numpy(), exM.cpu().numpy()
            for i, v in enumerate(views):
                s_ex = spearmanr(dn[i], exn[i])[0]
                s_exM = spearmanr(dn[i], exMn[i])[0]
                trow.append(dict(scene=spec, model=model, view=int(v), pearson_excess=p_ex[i], pearson_excess_mean_image=p_exM[i],
                                 pearson_plain_error=p_raw[i], spearman_excess=s_ex, spearman_excess_mean_image=s_exM))
        print(spec, model, "done", flush=True)
        del A, B, Mn
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"pixel_{sd}.csv"), index=False)
    pd.DataFrame(trow).to_csv(os.path.join(OUT, f"tiles_{sd}.csv"), index=False)
if __name__ == "__main__":
    main()
