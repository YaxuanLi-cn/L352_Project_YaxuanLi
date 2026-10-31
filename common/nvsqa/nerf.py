import itertools
import math
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from . import hashgrid_cuda
@dataclass


class NerfConfig:
    max_steps: int = 30_000
    rays_per_batch: int = 4096
    near: float = 0.05
    far: float = 1000.0
    num_proposal_samples: tuple = (256, 96)
    num_nerf_samples: int = 48
    proposal_update_every: int = 5
    proposal_warmup: int = 5000
    anneal_slope: float = 10.0
    anneal_max_iters: int = 1000
    interlevel_loss_mult: float = 1.0
    distortion_loss_mult: float = 0.002
    lr: float = 1e-2
    lr_final: float = 1e-4
    lr_decay_steps: int = 200_000
    num_levels: int = 16
    base_res: int = 16
    max_res: int = 2048
    log2_hashmap_size: int = 19
    hidden_dim: int = 64
    geo_feat_dim: int = 15
    eval_chunk: int = 1 << 15
    amp: bool = True
    seed: int = 42


class HashEncoding(nn.Module):
    PRIMES = (1, 2654435761, 805459861)
    def __init__(self, num_levels=16, min_res=16, max_res=2048, log2_hashmap_size=19, features=2):
        super().__init__()
        self.num_levels, self.features = num_levels, features
        self.table_size = 2 ** log2_hashmap_size
        growth = math.exp((math.log(max_res) - math.log(min_res)) / max(num_levels - 1, 1))
        res = torch.floor(min_res * growth ** torch.arange(num_levels, dtype=torch.float64)).long()
        self.register_buffer("res", res.float(), persistent=False)
        self.register_buffer("stride", res + 2, persistent=False)
        self.register_buffer("dense", (res + 2) ** 3 <= self.table_size, persistent=False)
        self.register_buffer("offset", torch.arange(num_levels) * self.table_size, persistent=False)
        corners = torch.tensor(list(itertools.product((0, 1), repeat=3)), dtype=torch.long)
        self.register_buffer("corners", corners, persistent=False)
        self.register_buffer("stride_i32", (res + 2).int(), persistent=False)
        self.register_buffer("dense_i32", ((res + 2) ** 3 <= self.table_size).int(), persistent=False)
        self.use_cuda_kernel = features == 2
        self.table = nn.Parameter(torch.empty(num_levels * self.table_size, features).uniform_(-1e-4, 1e-4))
    @property
    def out_dim(self):
        return self.num_levels * self.features
    def forward(self, x):
        x = x.float().clamp(0.0, 1.0)
        if self.use_cuda_kernel and x.is_cuda:
            return self._forward_cuda(x)
        return self._forward_torch(x)
    def _forward_cuda(self, x):
        return hashgrid_cuda.hashgrid(x, self.table, self.res, self.stride_i32, self.dense_i32, self.table_size)
    def _forward_torch(self, x):
        n = x.shape[0]
        xs = x[:, None, :].float() * self.res[None, :, None]
        x0 = torch.floor(xs)
        frac = xs - x0
        x0 = x0.long()
        c = x0[:, :, None, :] + self.corners[None, None]
        cx, cy, cz = c.unbind(-1)
        hashed = (cx * self.PRIMES[0]) ^ (cy * self.PRIMES[1]) ^ (cz * self.PRIMES[2])
        hashed = hashed & (self.table_size - 1)
        s = self.stride[None, :, None]
        dense = (cx + cy * s + cz * s * s).clamp_(0, self.table_size - 1)
        idx = torch.where(self.dense[None, :, None], dense, hashed) + self.offset[None, :, None]
        wc = torch.where(self.corners[None, None].bool(), frac[:, :, None, :], 1 - frac[:, :, None, :])
        w = wc.prod(-1)
        out = F.embedding_bag(idx.view(-1, 8), self.table, per_sample_weights=w.view(-1, 8), mode="sum")
        return out.view(n, self.num_levels * self.features)


def sh_encoding(d):
    x, y, z = d.unbind(-1)
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return torch.stack([
        torch.full_like(x, 0.28209479177387814),
        0.4886025119029199 * y, 0.4886025119029199 * z, 0.4886025119029199 * x,
        1.0925484305920792 * xy, 1.0925484305920792 * yz, 0.9461746957575601 * zz - 0.31539156525252,
        1.0925484305920792 * xz, 0.5462742152960396 * (xx - yy),
        0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * xy * z,
        0.4570457994644658 * y * (5 * zz - 1), 0.3731763325901154 * z * (5 * zz - 3),
        0.4570457994644658 * x * (5 * zz - 1), 1.445305721320277 * z * (xx - yy),
        0.5900435899266435 * x * (xx - 3 * yy),
    ], -1)


def trunc_exp(x):
    return torch.exp(x.float().clamp(max=15.0))


def contract(x):
    mag = x.abs().amax(-1, keepdim=True)
    return torch.where(mag <= 1, x, (2 - 1 / mag.clamp_min(1e-9)) * (x / mag.clamp_min(1e-9)))


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DensityField(nn.Module):
    def __init__(self, num_levels=5, max_res=128, log2_hashmap_size=17, hidden_dim=16):
        super().__init__()
        self.enc = HashEncoding(num_levels, 16, max_res, log2_hashmap_size, 2)
        self.mlp = mlp([self.enc.out_dim, hidden_dim, 1])
    def forward(self, x):
        xn = (contract(x) + 2.0) / 4.0
        return trunc_exp(self.mlp(self.enc(xn)).float() - 1.0)


class RadianceField(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = HashEncoding(cfg.num_levels, cfg.base_res, cfg.max_res, cfg.log2_hashmap_size, 2)
        self.mlp_base = mlp([self.enc.out_dim, cfg.hidden_dim, 1 + cfg.geo_feat_dim])
        self.mlp_head = mlp([16 + cfg.geo_feat_dim, cfg.hidden_dim, cfg.hidden_dim, 3])
    def forward(self, x, d):
        xn = (contract(x) + 2.0) / 4.0
        h = self.mlp_base(self.enc(xn))
        density = trunc_exp(h[..., :1].float() - 1.0)
        rgb = torch.sigmoid(self.mlp_head(torch.cat([sh_encoding(d).to(h.dtype), h[..., 1:]], -1)).float())
        return density, rgb


def spacing_fn(t):
    return torch.where(t < 1, t / 2, 1 - 1 / (2 * t))


def spacing_fn_inv(s):
    return torch.where(s < 0.5, 2 * s, 1 / (2 - 2 * s))


def render_weights(density, deltas):
    dd = density[..., 0] * deltas
    alphas = 1 - torch.exp(-dd)
    trans = torch.exp(-torch.cumsum(torch.cat([torch.zeros_like(dd[:, :1]), dd[:, :-1]], -1), -1))
    return torch.nan_to_num(alphas * trans)


def sample_pdf(s_bins, weights, num_samples, stratified, padding=0.01):
    k = weights.shape[-1]
    w = weights + padding
    w_sum = w.sum(-1, keepdim=True)
    fix = torch.relu(1e-5 - w_sum)
    w = w + fix / k
    w_sum = w_sum + fix
    cdf = torch.cumsum(w / w_sum, -1).clamp(max=1.0)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    nb = num_samples + 1
    u = torch.linspace(0.0, 1.0 - 1.0 / nb, nb, device=cdf.device)[None].expand(cdf.shape[0], nb)
    if stratified:
        u = u + torch.rand((cdf.shape[0], 1), device=cdf.device) / nb
    else:
        u = u + 1.0 / (2 * nb)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(0, k)
    above = inds.clamp(0, k)
    cdf0, cdf1 = cdf.gather(1, below), cdf.gather(1, above)
    b0, b1 = s_bins.gather(1, below), s_bins.gather(1, above)
    t = torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), nan=0.0).clamp(0, 1)
    return (b0 + t * (b1 - b0)).detach()


def outer(t0_starts, t0_ends, t1_starts, t1_ends, y1):
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, -1)], -1)
    n = t1_starts.shape[-1]
    idx_lo = (torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), right=True) - 1).clamp(0, n - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), right=True).clamp(0, n - 1)
    return cy1[..., 1:].gather(-1, idx_hi) - cy1[..., :-1].gather(-1, idx_lo)


def interlevel_loss(s_fine, w_fine, s_prop_list, w_prop_list):
    w = w_fine.detach()
    c = s_fine.detach()
    loss = 0.0
    for sp, wp in zip(s_prop_list, w_prop_list):
        w_outer = outer(c[..., :-1], c[..., 1:], sp[..., :-1], sp[..., 1:], wp)
        loss = loss + (torch.clip(w - w_outer, min=0) ** 2 / (w + 1e-7)).mean()
    return loss


def distortion_loss(s, w):
    mid = (s[..., 1:] + s[..., :-1]) / 2
    inter = (w * (w[..., None, :] * (mid[..., :, None] - mid[..., None, :]).abs()).sum(-1)).sum(-1)
    intra = (w ** 2 * (s[..., 1:] - s[..., :-1])).sum(-1) / 3
    return (inter + intra).mean()


class NerfModel(nn.Module):
    def __init__(self, cfg, transform):
        super().__init__()
        self.cfg = cfg
        self.field = RadianceField(cfg)
        self.props = nn.ModuleList([DensityField(5, 128, 17, 16), DensityField(5, 256, 17, 16)])
        self.register_buffer("transform", torch.as_tensor(transform, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(float(np.linalg.norm(np.asarray(transform)[0, :3]))))
        self.step = 0
        self._since_update = 0
    def rays_from_pixels(self, c2w, K, px, py):
        d_cam = torch.stack([(px + 0.5 - K[:, 0, 2]) / K[:, 0, 0], (py + 0.5 - K[:, 1, 2]) / K[:, 1, 1],
                             torch.ones_like(px)], -1)
        rot = self.transform[:3, :3] / self.scale
        d = torch.einsum("ij,rjk,rk->ri", rot, c2w[:, :3, :3], d_cam)
        o = torch.einsum("ij,rj->ri", self.transform[:3, :3], c2w[:, :3, 3]) + self.transform[:3, 3]
        return o, F.normalize(d, dim=-1)
    def render_rays(self, o, d, training):
        cfg = self.cfg
        nr = o.shape[0]
        s_near = spacing_fn(torch.tensor(cfg.near, device=o.device))
        s_far = spacing_fn(torch.tensor(cfg.far, device=o.device))
        def s_to_t(s):
            return spacing_fn_inv(s * s_far + (1 - s) * s_near)
        update = True
        anneal = 1.0
        if training:
            sched = float(np.clip(np.interp(self.step, [0, cfg.proposal_warmup], [0, cfg.proposal_update_every]),
                                  1, cfg.proposal_update_every))
            self._since_update += 1
            update = self._since_update > sched or self.step < 10
            if update:
                self._since_update = 0
            frac = min(max(self.step / cfg.anneal_max_iters, 0.0), 1.0)
            anneal = (cfg.anneal_slope * frac) / ((cfg.anneal_slope - 1) * frac + 1)
        n0 = cfg.num_proposal_samples[0]
        s_bins = torch.linspace(0, 1, n0 + 1, device=o.device)[None].expand(nr, n0 + 1)
        if training:
            centers = (s_bins[:, 1:] + s_bins[:, :-1]) / 2
            upper = torch.cat([centers, s_bins[:, -1:]], -1)
            lower = torch.cat([s_bins[:, :1], centers], -1)
            s_bins = lower + (upper - lower) * torch.rand((nr, 1), device=o.device)
        s_list, w_list = [], []
        num_next = list(cfg.num_proposal_samples[1:]) + [cfg.num_nerf_samples]
        for i, prop in enumerate(self.props):
            t_bins = s_to_t(s_bins)
            mid = (t_bins[:, 1:] + t_bins[:, :-1]) / 2
            pos = o[:, None] + d[:, None] * mid[..., None]
            with torch.set_grad_enabled(training and update):
                density = prop(pos.reshape(-1, 3)).view(nr, -1, 1)
                w = render_weights(density, t_bins[:, 1:] - t_bins[:, :-1])
            s_list.append(s_bins)
            w_list.append(w)
            s_bins = sample_pdf(s_bins, torch.pow(w.detach(), anneal), num_next[i], stratified=training)
        t_bins = s_to_t(s_bins)
        mid = (t_bins[:, 1:] + t_bins[:, :-1]) / 2
        pos = o[:, None] + d[:, None] * mid[..., None]
        ns = mid.shape[1]
        density, rgb = self.field(pos.reshape(-1, 3), d[:, None].expand(nr, ns, 3).reshape(-1, 3))
        density, rgb = density.view(nr, ns, 1), rgb.view(nr, ns, 3)
        w = render_weights(density, t_bins[:, 1:] - t_bins[:, :-1])
        acc = w.sum(-1, keepdim=True)
        comp = (w[..., None] * rgb).sum(1) + (1 - acc) * rgb[:, -1]
        out = {"rgb": comp, "acc": acc}
        if training:
            out.update(s_fine=s_bins, w_fine=w, s_props=s_list, w_props=w_list, props_updated=update)
        return out
    @torch.no_grad()
    def render(self, c2w, K, width, height):
        self.eval()
        dev = self.transform.device
        py, px = torch.meshgrid(torch.arange(height, device=dev, dtype=torch.float32),
                                torch.arange(width, device=dev, dtype=torch.float32), indexing="ij")
        px, py = px.reshape(-1), py.reshape(-1)
        out = []
        for i in range(0, px.numel(), self.cfg.eval_chunk):
            sl = slice(i, i + self.cfg.eval_chunk)
            n = px[sl].numel()
            o, d = self.rays_from_pixels(c2w[None].expand(n, 4, 4), K[None].expand(n, 3, 3), px[sl], py[sl])
            with torch.autocast("cuda", enabled=self.cfg.amp):
                out.append(self.render_rays(o, d, training=False)["rgb"].float())
        img = torch.cat(out).view(height, width, 3)
        return (img.clamp(0, 1) * 255 + 0.5).byte()


def scene_transform(c2w_train):
    c2w = np.asarray(c2w_train, np.float64)
    centre = c2w[:, :3, 3].mean(0)
    up = -c2w[:, :3, 1].mean(0)
    up = up / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    c = float(np.dot(up, z))
    if np.linalg.norm(v) < 1e-8:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx / (1 + c)
    pos = (c2w[:, :3, 3] - centre) @ rot.T
    scale = 1.0 / np.abs(pos).max()
    m = np.eye(4)
    m[:3, :3] = scale * rot
    m[:3, 3] = -scale * rot @ centre
    return m


def train_nerf(scene, train_ids, cfg, snapshot_steps=(), snapshot_fn=None, device="cuda", log=print,
               log_every=2500):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(device)
    train_ids = np.asarray(train_ids)
    images = scene.images[train_ids].to(dev)
    c2w = torch.from_numpy(scene.c2w[train_ids]).float().to(dev)
    K = torch.from_numpy(scene.K[train_ids]).float().to(dev)
    n, H, W = images.shape[:3]
    model = NerfModel(cfg, scene_transform(scene.c2w[train_ids])).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-15)
    gamma = (cfg.lr_final / cfg.lr) ** (1.0 / cfg.lr_decay_steps)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    snapshot_steps = set(int(s) for s in snapshot_steps)
    t0 = time.time()
    t_snap = 0.0
    for step in range(cfg.max_steps + 1):
        if step in snapshot_steps and snapshot_fn is not None:
            ts = time.time()
            snapshot_fn(step, model, {"train_time": time.time() - t0 - t_snap})
            t_snap += time.time() - ts
        if step == cfg.max_steps:
            break
        model.train()
        model.step = step
        idx = torch.randint(0, n, (cfg.rays_per_batch,), device=dev)
        py = torch.randint(0, H, (cfg.rays_per_batch,), device=dev)
        px = torch.randint(0, W, (cfg.rays_per_batch,), device=dev)
        gt = images[idx, py, px].float() / 255.0
        o, d = model.rays_from_pixels(c2w[idx], K[idx], px.float(), py.float())
        with torch.autocast("cuda", enabled=cfg.amp):
            out = model.render_rays(o, d, training=True)
        loss_rgb = F.mse_loss(out["rgb"].float(), gt)
        loss = loss_rgb + cfg.distortion_loss_mult * distortion_loss(out["s_fine"], out["w_fine"])
        if out["props_updated"]:
            loss = loss + cfg.interlevel_loss_mult * interlevel_loss(out["s_fine"], out["w_fine"],
                                                                     out["s_props"], out["w_props"])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if log_every and (step + 1) % log_every == 0:
            log(f"  [nerf] step {step + 1}/{cfg.max_steps} mse {loss_rgb.item():.5f} "
                f"psnr {-10 * math.log10(loss_rgb.item()):.2f} {time.time() - t0 - t_snap:.0f}s")
    return model, {"train_time": time.time() - t0 - t_snap}
