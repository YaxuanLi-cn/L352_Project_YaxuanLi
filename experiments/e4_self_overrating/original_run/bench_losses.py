#!/usr/bin/env python
import json, os, sys, time
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import torch
torch.backends.cudnn.benchmark = True
from gh import metrics as M
from gh import train as T
H, W = int(sys.argv[1]), int(sys.argv[2])
m = M.Metrics()
gt = torch.rand(1, 3, H, W, device="cuda")
res = {}
for name in ["default"] + list(M.LOSSES):
    x = (gt + 0.05 * torch.randn_like(gt)).clamp(0, 1).requires_grad_(True)
    f = (lambda p, g: T.default_loss(p, g)) if name == "default" else (lambda p, g, n=name: m.loss(n, p, g))
    for _ in range(3):
        f(x, gt).backward()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(10):
        x.grad = None
        f(x, gt).backward()
    torch.cuda.synchronize()
    res[name] = dict(ms=(time.time() - t0) * 100, peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    print(name, f"{res[name]['ms']:.1f} ms  {res[name]['peak_gb']:.2f} GB", flush=True)
json.dump(res, open(os.path.join(P.ORIG_RESULTS, f"bench_{H}x{W}.json"), "w"), indent=1)
