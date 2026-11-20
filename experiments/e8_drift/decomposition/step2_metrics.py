#!/usr/bin/env python
import os
import sys
import time
import pandas as pd
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib as L              
from gh import metrics as M              
torch.backends.cudnn.benchmark = True
OUT = L.OUT
FIRST = [L.CTRL, "lpips_vgg__pure", "lpips_alex__pure", "dists__pure", "lpips_vgg__mixed", "lpips_alex__mixed", "dists__mixed",
         "cvvdp__pure", "flip__pure", "l1__pure", "msssim__pure"]
@torch.no_grad()


def main():
    spec, nmax, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    sd = spec.replace("/", "_")
    f = os.path.join(OUT, f"metrics_{sd}.csv")
    views = L.pick(L.held_out(spec), nmax)
    m = M.Metrics()
    G = L.reference(spec, views)
    rows = []
    t0 = time.time()
    order = FIRST + [x for x in L.models() if x not in FIRST]
    for model in order:
        A8, B8 = L.pair(spec, model, views)
        for i, v in enumerate(views):
            g = torch.from_numpy(G[i:i + 1]).cuda().permute(0, 3, 1, 2).float() / 255.0
            a = torch.from_numpy(A8[i:i + 1]).cuda().permute(0, 3, 1, 2).float() / 255.0
            b = torch.from_numpy(B8[i:i + 1]).cuda().permute(0, 3, 1, 2).float() / 255.0
            mean = 0.5 * (a + b)
            tags, x = (("A", "B", "M"), torch.cat([a, b, mean])) if mode == "abm" else (("M",), mean)
            gg = g.expand(x.shape[0], -1, -1, -1).contiguous()
            for n in M.IMAGE_METRICS:
                val = m(n, x, gg).reshape(-1).tolist()
                for tag, y in zip(tags, val):
                    rows.append(dict(scene=spec, model=model, image=tag, view=int(v), metric=n, value=float(y)))
        print(f"{spec} {model} done {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f, index=False)
if __name__ == "__main__":
    main()
