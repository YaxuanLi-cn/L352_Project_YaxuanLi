import concurrent.futures as cf
import multiprocessing as mp
import os
import numpy as np
import torch
import torch.nn.functional as F
IMAGE_METRICS = ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg", "flip", "cvvdp_image"]
VIDEO_METRICS = ["cvvdp", "cvvdp_sust"]
EXTRA_IMAGE_METRICS = ["stlpips_alex", "stlpips_vgg"]
SIGN = {"psnr": 1, "ssim": 1, "ms_ssim": 1, "lpips_alex": -1, "lpips_vgg": -1, "flip": -1,
        "cvvdp_image": 1, "cvvdp": 1, "cvvdp_sust": 1, "stlpips_alex": -1, "stlpips_vgg": -1}
LABEL = {"psnr": "PSNR", "ssim": "SSIM", "ms_ssim": "MS-SSIM", "lpips_alex": "LPIPS (Alex)",
         "lpips_vgg": "LPIPS (VGG)", "flip": "FLIP", "cvvdp_image": "ColorVideoVDP (image)",
         "cvvdp": "ColorVideoVDP (video)", "cvvdp_sust": "ColorVideoVDP (video, no transient ch.)",
         "stlpips_alex": "ST-LPIPS (Alex)", "stlpips_vgg": "ST-LPIPS (VGG)"}
DISPLAYS = {
    "fhd_3h": ("standard_fhd", dict(resolution=[1920, 1080], distance_display_heights=3, diagonal_size_inches=24)),
    "fhd_1.5h": ("standard_fhd", dict(resolution=[1920, 1080], distance_display_heights=1.5, diagonal_size_inches=24)),
    "fhd_6h": ("standard_fhd", dict(resolution=[1920, 1080], distance_display_heights=6, diagonal_size_inches=24)),
    "standard_fhd": ("standard_fhd", None),
    "standard_4k": ("standard_4k", None),
}


def _gauss(size, sigma, device):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def ssim(x, y, size=11, sigma=1.5, c1=0.01 ** 2, c2=0.03 ** 2):
    c = x.shape[1]
    g = _gauss(size, sigma, x.device)
    kh = g.view(1, 1, 1, size).expand(c, 1, 1, size)
    kv = g.view(1, 1, size, 1).expand(c, 1, size, 1)
    def blur(t):
        return F.conv2d(F.conv2d(t, kh, groups=c), kv, groups=c)
    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x ** 2
    syy = blur(y * y) - mu_y ** 2
    sxy = blur(x * y) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean(dim=(1, 2, 3))


def _flip_worker(args):
    import flip_evaluator
    ref, test, ppd = args
    _, mean, _ = flip_evaluator.evaluate(np.ascontiguousarray(ref, dtype=np.float32) / 255.0,
                                         np.ascontiguousarray(test, dtype=np.float32) / 255.0,
                                         "LDR", inputsRGB=True, applyMagma=False, computeMeanError=True,
                                         parameters={"ppd": float(ppd)})
    return float(mean)


class MetricSuite:
    def __init__(self, device="cuda", display="fhd_3h", metrics=None, flip_procs=10, flip_threads=2,
                 cvvdp_gpu_mem=4):
        self.device = torch.device(device)
        self.display = display
        self.names = list(metrics) if metrics else IMAGE_METRICS + VIDEO_METRICS
        self._lpips = {}
        self._cvvdp = None
        self._pool = None
        self.flip_procs, self.flip_threads = flip_procs, flip_threads
        self.cvvdp_gpu_mem = cvvdp_gpu_mem
        import pycvvdp
        photometry, geom = DISPLAYS[display]
        self._geom = pycvvdp.vvdp_display_geometry(**geom) if geom else pycvvdp.vvdp_display_geometry.load(photometry)
        self._photometry = photometry
        self.ppd = float(self._geom.get_ppd())
    def lpips_net(self, net):
        if net not in self._lpips:
            import lpips
            self._lpips[net] = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()
        return self._lpips[net]
    def stlpips_net(self, net):
        key = "st_" + net
        if key not in self._lpips:
            from stlpips_pytorch import stlpips
            self._lpips[key] = stlpips.LPIPS(net=net, variant="shift_tolerant", verbose=False).to(self.device).eval()
        return self._lpips[key]
    @property
    def cvvdp(self):
        if self._cvvdp is None:
            import pycvvdp
            self._cvvdp = pycvvdp.cvvdp(display_name=self._photometry, display_geometry=self._geom, quiet=True,
                                        device=self.device, gpu_mem=self.cvvdp_gpu_mem)
        return self._cvvdp
    @property
    def pool(self):
        if self._pool is None:
            old = os.environ.get("OMP_NUM_THREADS")
            os.environ["OMP_NUM_THREADS"] = str(self.flip_threads)
            self._pool = cf.ProcessPoolExecutor(self.flip_procs, mp_context=mp.get_context("spawn"))
            list(self._pool.map(abs, range(self.flip_procs)))
            if old is None:
                os.environ.pop("OMP_NUM_THREADS")
            else:
                os.environ["OMP_NUM_THREADS"] = old
        return self._pool
    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
    @torch.no_grad()
    def images(self, test, ref, batch=None, names=None):
        names = [n for n in (names or self.names) if n in IMAGE_METRICS + EXTRA_IMAGE_METRICS]
        test = torch.as_tensor(test)
        ref = torch.as_tensor(ref)
        n = test.shape[0]
        if batch is None:
            batch = max(1, min(8, int(2.0e6 // (test.shape[1] * test.shape[2]))))
        out = {k: [] for k in names}
        flip_futs = None
        if "flip" in names:
            tn, rn = test.cpu().numpy(), ref.cpu().numpy()
            flip_futs = [self.pool.submit(_flip_worker, (rn[i], tn[i], self.ppd)) for i in range(n)]
        for i in range(0, n, batch):
            t8 = test[i:i + batch].to(self.device)
            r8 = ref[i:i + batch].to(self.device)
            t = t8.permute(0, 3, 1, 2).float() / 255.0
            r = r8.permute(0, 3, 1, 2).float() / 255.0
            if "psnr" in names:
                mse = ((t - r) ** 2).mean(dim=(1, 2, 3))
                out["psnr"].append(-10 * torch.log10(mse.clamp_min(1e-12)))
            if "ssim" in names:
                out["ssim"].append(ssim(t, r))
            if "ms_ssim" in names:
                from pytorch_msssim import ms_ssim
                out["ms_ssim"].append(ms_ssim(t, r, data_range=1.0, size_average=False))
            if "lpips_alex" in names:
                out["lpips_alex"].append(self.lpips_net("alex")(t, r, normalize=True).flatten())
            if "lpips_vgg" in names:
                out["lpips_vgg"].append(self.lpips_net("vgg")(t, r, normalize=True).flatten())
            for net in ("alex", "vgg"):
                if "stlpips_" + net in names:
                    out["stlpips_" + net].append(self.stlpips_net(net)(t, r, normalize=True).flatten())
            if "cvvdp_image" in names:
                jod, _ = self.cvvdp.predict(t8, r8, dim_order="BHWC")
                out["cvvdp_image"].append(jod.reshape(-1))
        res = {k: torch.cat(v).double().cpu().numpy() for k, v in out.items() if k != "flip"}
        if flip_futs is not None:
            res["flip"] = np.array([f.result() for f in flip_futs])
        return res
    @torch.no_grad()
    def video(self, test, ref, fps, names=None, image_metrics=True):
        names = names or self.names
        test = torch.as_tensor(test)
        ref = torch.as_tensor(ref)
        scores, frames = {}, {}
        if image_metrics:
            frames.update(self.images(test, ref, names=names))
            scores.update({k: float(np.mean(v)) for k, v in frames.items()})
        if "cvvdp" in names or "cvvdp_sust" in names:
            m = self.cvvdp
            jod, stats = m.predict(test.to(self.device), ref.to(self.device), dim_order="FHWC",
                                   frames_per_second=float(fps))
            q = torch.as_tensor(stats["Q_per_ch"], device=self.device)
            scores["cvvdp"] = float(jod)
            scores["cvvdp_sust"] = float(m.do_pooling_and_jods(q[:, :3]))
            frames["cvvdp_q_per_ch"] = self.pool_bands(q)[0].cpu().numpy()
            del q, stats
        torch.cuda.empty_cache()
        return scores, frames
    def pool_bands(self, q):
        m = self.cvvdp
        nch, nb = q.shape[1], q.shape[3]
        per_ch_w = m.get_ch_weights(nch)
        per_sband_w = torch.ones((1, nch, 1, nb), dtype=torch.float32, device=q.device)
        per_sband_w[:, :, 0, -1] = m.baseband_weight[0:nch]
        return m.lp_norm(q * per_ch_w * per_sband_w, m.beta_sch, dim=3, normalize=False)[..., 0]
