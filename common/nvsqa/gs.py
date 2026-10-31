import math
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from scipy.spatial import cKDTree
C0 = 0.28209479177387814
@dataclass


class GSConfig:
    strategy: str = "default"
    cap_max: int = 1_000_000
    max_steps: int = 30_000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    near: float = 0.01
    far: float = 1e10
    seed: int = 42
    @staticmethod
    def mcmc(cap_max, **kw):
        return GSConfig(strategy="mcmc", cap_max=cap_max, init_opa=0.5, init_scale=0.1,
                        opacity_reg=0.01, scale_reg=0.01, **kw)


def _gauss_kernel(size=11, sigma=1.5, device="cpu"):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def ssim_map(x, y, size=11, sigma=1.5, c1=0.01 ** 2, c2=0.03 ** 2):
    c = x.shape[1]
    g = _gauss_kernel(size, sigma, x.device)
    kh = g.view(1, 1, 1, size).expand(c, 1, 1, size)
    kv = g.view(1, 1, size, 1).expand(c, 1, size, 1)
    pad = size // 2
    def blur(t):
        t = F.conv2d(t, kh, padding=(0, pad), groups=c)
        return F.conv2d(t, kv, padding=(pad, 0), groups=c)
    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x ** 2
    syy = blur(y * y) - mu_y ** 2
    sxy = blur(x * y) - mu_x * mu_y
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


class GaussianModel:
    def __init__(self, params, cfg, active_sh_degree=None):
        self.params = params
        self.cfg = cfg
        self.active_sh_degree = cfg.sh_degree if active_sh_degree is None else active_sh_degree
    @property
    def num_gaussians(self):
        return self.params["means"].shape[0]
    def rasterize(self, viewmats, Ks, width, height, sh_degree=None, **kw):
        p = self.params
        colors = torch.cat([p["sh0"], p["shN"]], 1)
        return rasterization(
            means=p["means"], quats=p["quats"], scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]), colors=colors, viewmats=viewmats, Ks=Ks,
            width=width, height=height, sh_degree=self.active_sh_degree if sh_degree is None else sh_degree,
            near_plane=self.cfg.near, far_plane=self.cfg.far, packed=False, **kw)
    @torch.no_grad()
    def render(self, c2w, K, width, height):
        viewmat = torch.linalg.inv(c2w)[None]
        img, _, _ = self.rasterize(viewmat, K[None], width, height)
        return (img[0].clamp(0, 1) * 255 + 0.5).byte()
    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.params.items()}


def init_gaussians(points, colors, cfg, scene_scale, device):
    points = torch.as_tensor(points, dtype=torch.float32)
    rgb = torch.as_tensor(colors, dtype=torch.float32) / 255.0
    if cfg.strategy == "mcmc" and len(points) > cfg.cap_max:
        g = torch.Generator().manual_seed(cfg.seed)
        keep = torch.randperm(len(points), generator=g)[:cfg.cap_max]
        points, rgb = points[keep], rgb[keep]
    d, _ = cKDTree(points.numpy()).query(points.numpy(), k=4)
    dist = torch.from_numpy(np.sqrt((d[:, 1:] ** 2).mean(1))).float().clamp_min(1e-7)
    n = len(points)
    scales = torch.log(dist * cfg.init_scale)[:, None].repeat(1, 3)
    g = torch.Generator().manual_seed(cfg.seed)
    quats = torch.rand((n, 4), generator=g)
    opacities = torch.logit(torch.full((n,), cfg.init_opa))
    k = (cfg.sh_degree + 1) ** 2
    sh = torch.zeros((n, k, 3))
    sh[:, 0] = (rgb - 0.5) / C0
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh[:, :1].contiguous()),
        "shN": torch.nn.Parameter(sh[:, 1:].contiguous()),
    }).to(device)
    lrs = {"means": cfg.means_lr * scene_scale, "scales": cfg.scales_lr, "quats": cfg.quats_lr,
           "opacities": cfg.opacities_lr, "sh0": cfg.sh0_lr, "shN": cfg.shN_lr}
    optimizers = {name: torch.optim.Adam([{"params": params[name], "lr": lr, "name": name}], eps=1e-15)
                  for name, lr in lrs.items()}
    return params, optimizers


def train_gs(scene, train_ids, cfg, snapshot_steps=(), snapshot_fn=None, device="cuda", log=print,
             log_every=2500):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(device)
    train_ids = np.asarray(train_ids)
    images = scene.images[train_ids].to(dev)
    viewmats = torch.from_numpy(np.linalg.inv(scene.c2w[train_ids])).float().to(dev)
    Ks = torch.from_numpy(scene.K[train_ids]).float().to(dev)
    W, H = scene.width, scene.height
    scene_scale = scene.camera_extent(train_ids) * 1.1
    params, optimizers = init_gaussians(scene.points, scene.points_rgb, cfg, scene_scale, dev)
    model = GaussianModel(params, cfg, active_sh_degree=0)
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps))
    if cfg.strategy == "default":
        strategy = DefaultStrategy(verbose=False)
    else:
        strategy = MCMCStrategy(cap_max=cfg.cap_max, verbose=False)
    state = strategy.initialize_state(scene_scale=scene_scale) if cfg.strategy == "default" \
        else strategy.initialize_state()                  
    snapshot_steps = set(int(s) for s in snapshot_steps)
    g = torch.Generator().manual_seed(cfg.seed)
    order = torch.randperm(len(train_ids), generator=g).tolist()
    t0 = time.time()
    t_snap = 0.0
    for step in range(cfg.max_steps + 1):
        if step in snapshot_steps and snapshot_fn is not None:
            ts = time.time()
            model.active_sh_degree = min(max(step - 1, 0) // cfg.sh_degree_interval, cfg.sh_degree)
            snapshot_fn(step, model, {"train_time": time.time() - t0 - t_snap,
                                      "num_gaussians": model.num_gaussians})
            t_snap += time.time() - ts
        if step == cfg.max_steps:
            break
        if not order:
            order = torch.randperm(len(train_ids), generator=g).tolist()
        i = order.pop()
        gt = images[i].float() / 255.0
        sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        renders, alphas, info = model.rasterize(viewmats[i:i + 1], Ks[i:i + 1], W, H, sh_degree=sh_degree)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        pred = renders[0]
        l1 = (pred - gt).abs().mean()
        ssim = ssim_map(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).mean()
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim)
        if cfg.opacity_reg > 0:
            loss = loss + cfg.opacity_reg * torch.sigmoid(params["opacities"]).mean()
        if cfg.scale_reg > 0:
            loss = loss + cfg.scale_reg * torch.exp(params["scales"]).mean()
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        if cfg.strategy == "default":
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, lr=sched.get_last_lr()[0])
        if log_every and (step + 1) % log_every == 0:
            log(f"  [3dgs] step {step + 1}/{cfg.max_steps} loss {loss.item():.4f} "
                f"N={model.num_gaussians} {time.time() - t0 - t_snap:.0f}s")
    model.active_sh_degree = cfg.sh_degree
    return model, {"train_time": time.time() - t0 - t_snap, "num_gaussians": model.num_gaussians}
