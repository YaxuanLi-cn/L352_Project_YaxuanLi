import math
import os
import time
import numpy as np
import torch
from nvsqa import gs as nvsqa_gs
from . import metrics as M


def stage_a(scene, seed, ckpt_path, steps=30_000, log=print, half=False):
    cfg = nvsqa_gs.GSConfig(seed=seed, max_steps=steps)
    model, info = nvsqa_gs.train_gs(scene, scene.train_ids, cfg, device="cuda", log=log, log_every=5000)
    sd = model.state_dict()
    if half:
        sd = {k: (v if k == "means" else v.half()) for k, v in sd.items()}
    ck = dict(params=sd, seed=seed, steps=steps, scene_scale=scene.camera_extent(scene.train_ids) * 1.1,
              num_gaussians=int(info["num_gaussians"]), train_time=float(info["train_time"]),
              cfg=cfg.__dict__)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(ck, ckpt_path + ".tmp")
    os.replace(ckpt_path + ".tmp", ckpt_path)
    del model
    torch.cuda.empty_cache()
    return ck


def load_model(ck, device="cuda", requires_grad=False):
    cfg = nvsqa_gs.GSConfig(**ck["cfg"])
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.float().clone(), requires_grad=requires_grad)
                                     for k, v in ck["params"].items()}).to(device)
    return nvsqa_gs.GaussianModel(params, cfg, active_sh_degree=cfg.sh_degree)


def default_loss(pred, gt, lam=0.2):
    l1 = (pred - gt).abs().mean()
    ssim = nvsqa_gs.ssim_map(pred, gt).mean()
    return (1 - lam) * l1 + lam * (1 - ssim)


def make_loss(loss, variant, metrics, weight=None):
    if variant == "ctrl":
        return lambda p, g: default_loss(p, g)
    if variant == "pure":
        return lambda p, g: metrics.loss(loss, p, g)
    if variant.startswith("mixed"):
        return lambda p, g: default_loss(p, g) + weight * metrics.loss(loss, p, g)
@torch.no_grad()


def mixed_weight(ck, scene, view_ids, loss, metrics, device="cuda"):
    model = load_model(ck, device)
    d, m = [], []
    for i in view_ids:
        pred, gt = render_train(model, scene, i, device)
        d.append(float(default_loss(pred, gt)))
        m.append(float(metrics.loss(loss, pred, gt)))
    del model
    torch.cuda.empty_cache()
    return float(np.mean(d) / np.mean(m)), float(np.mean(d)), float(np.mean(m))


def render_train(model, scene, i, device="cuda"):
    c2w = torch.from_numpy(scene.c2w[i]).float().to(device)
    K = torch.from_numpy(scene.K[i]).float().to(device)
    img, _, _ = model.rasterize(torch.linalg.inv(c2w)[None], K[None], scene.width, scene.height)
    gt = scene.images[i].to(device).float().permute(2, 0, 1)[None] / 255.0
    return img[0].permute(2, 0, 1)[None], gt


def stage_b(ck, scene, loss_fn, steps, seed, device="cuda", log=print, log_every=1000):
    torch.manual_seed(seed)
    cfg = nvsqa_gs.GSConfig(**ck["cfg"])
    model = load_model(ck, device, requires_grad=True)
    params = model.params
    scene_scale = ck["scene_scale"]
    lrs = {"means": cfg.means_lr * scene_scale * 0.01,
           "scales": cfg.scales_lr, "quats": cfg.quats_lr, "opacities": cfg.opacities_lr,
           "sh0": cfg.sh0_lr, "shN": cfg.shN_lr}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    train_ids = np.asarray(scene.train_ids)
    g = torch.Generator().manual_seed(1000 + seed)
    order = []
    curve = []
    status = "ok"
    t0 = time.time()
    acc = 0.0
    for step in range(steps):
        if not order:
            order = torch.randperm(len(train_ids), generator=g).tolist()
        i = int(train_ids[order.pop()])
        pred, gt = render_train(model, scene, i, device)
        loss = loss_fn(pred, gt)
        if not torch.isfinite(loss):
            status = f"diverged: non-finite loss at step {step}"
            break
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None):
            status = f"diverged: non-finite gradient at step {step}"
            for o in opts.values():
                o.zero_grad(set_to_none=True)
            break
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        acc += float(loss)
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                bias = (pred - gt).mean(dim=(0, 2, 3)).tolist()
            curve.append(dict(step=step + 1, loss=acc / 100, rgb_bias=bias))
            acc = 0.0
        if log_every and (step + 1) % log_every == 0:
            log(f"    step {step + 1}/{steps} loss {curve[-1]['loss']:.5f} {time.time() - t0:.0f}s")
    for p in params.values():
        p.requires_grad_(False)
    return model, dict(status=status, steps_done=step + 1 if status == "ok" else step, train_time=time.time() - t0,
                       curve=curve, num_gaussians=int(model.num_gaussians), lrs=lrs)
