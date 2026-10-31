import os
import torch
import paths as P


def load_params(path):
    path = path if os.path.isabs(path) else P.legacy(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["encoding"] == "fp32":
        params = {k: v.float() for k, v in ck["params"].items()}
    else:
        base = torch.load(P.legacy(ck["base"]), map_location="cpu", weights_only=False)
        params = {k: base["params"][k].float() + v.float() for k, v in ck["params"].items()}
    return dict(params=params, cfg=ck["cfg"], scene_scale=ck["scene_scale"], encoding=ck["encoding"])


def load(path, device="cuda"):
    from gh import train as T
    return T.load_model(load_params(path), device=device)
