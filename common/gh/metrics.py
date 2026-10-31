import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from nvsqa import metrics as nvsqa_metrics              
IMAGE_METRICS = ["l1", "mse", "psnr", "ssim", "ms_ssim", "lpips_vgg", "lpips_alex", "dists", "flip",
                 "cvvdp_image", "dreamsim", "fsim", "vif"]
JUDGES = ["dreamsim", "fsim", "vif"]
VIDEO_METRICS = ["cvvdp", "cvvdp_sust"]
SIGN = {"l1": -1, "mse": -1, "psnr": 1, "ssim": 1, "ms_ssim": 1, "lpips_vgg": -1, "lpips_alex": -1,
        "dists": -1, "flip": -1, "cvvdp_image": 1, "dreamsim": -1, "fsim": 1, "vif": 1,
        "cvvdp": 1, "cvvdp_sust": 1}
FAMILY = {"l1": "pixel", "mse": "pixel", "psnr": "pixel", "ssim": "ssim", "ms_ssim": "ssim",
          "lpips_vgg": "lpips", "lpips_alex": "lpips", "dists": "dists", "flip": "flip",
          "cvvdp_image": "cvvdp", "cvvdp": "cvvdp", "cvvdp_sust": "cvvdp",
          "dreamsim": "dreamsim", "fsim": "fsim", "vif": "vif"}
LABEL = {"l1": "L1", "mse": "MSE", "psnr": "PSNR", "ssim": "SSIM", "ms_ssim": "MS-SSIM",
         "lpips_vgg": "LPIPS-VGG", "lpips_alex": "LPIPS-Alex", "dists": "DISTS", "flip": "FLIP",
         "cvvdp_image": "CVVDP (image)", "cvvdp": "CVVDP (video)", "cvvdp_sust": "CVVDP (video, no transient)",
         "dreamsim": "DreamSim", "fsim": "FSIM", "vif": "VIF"}
LOSSES = {"l1": "l1", "l2": "mse", "dssim": "ssim", "msssim": "ms_ssim", "lpips_vgg": "lpips_vgg",
          "lpips_alex": "lpips_alex", "dists": "dists", "flip": "flip", "cvvdp": "cvvdp_image"}
LOSS_LABEL = {"default": "default (ctrl)", "l1": "L1", "l2": "L2", "dssim": "D-SSIM", "msssim": "MS-SSIM",
              "lpips_vgg": "LPIPS-VGG", "lpips_alex": "LPIPS-Alex", "dists": "DISTS", "flip": "FLIP",
              "cvvdp": "CVVDP"}
DISPLAY = "fhd_3h"


def loss_form(metric, value):
    if metric in ("l1", "mse", "lpips_vgg", "lpips_alex", "dists", "flip", "dreamsim"):
        return value
    if metric in ("ssim", "ms_ssim", "fsim", "vif"):
        return 1.0 - value
    if metric == "cvvdp_image":
        return 10.0 - value
    if metric == "psnr":
        return -value


class Metrics:
    def __init__(self, device="cuda", cache_dir=os.path.expanduser("~/.cache/goodhart")):
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self._nets = {}
        import pycvvdp
        photometry, geom = nvsqa_metrics.DISPLAYS[DISPLAY]
        self._geom = pycvvdp.vvdp_display_geometry(**geom)
        self._photometry = photometry
        self.ppd = float(self._geom.get_ppd())
    def _net(self, key):
        if key in self._nets:
            return self._nets[key]
        if key in ("lpips_vgg", "lpips_alex"):
            import lpips
            net = lpips.LPIPS(net=key.split("_")[1], verbose=False).to(self.device).eval()
        elif key == "dists":
            import DISTS_pytorch
            from DISTS_pytorch.DISTS_pt import DISTS
            net = DISTS(load_weights=False)
            w = torch.load(os.path.join(os.path.dirname(DISTS_pytorch.__file__), "weights.pt"), map_location="cpu")
            net.alpha.data, net.beta.data = w["alpha"], w["beta"]
            net = net.to(self.device).eval()
        elif key == "dreamsim":
            from dreamsim import dreamsim
            os.makedirs(self.cache_dir, exist_ok=True)
            hub = torch.hub.get_dir()
            net, _ = dreamsim(pretrained=True, device=str(self.device),
                              cache_dir=os.path.join(self.cache_dir, "dreamsim"))
            torch.hub.set_dir(hub)
            net = net.eval()
        elif key == "cvvdp":
            import pycvvdp
            net = pycvvdp.cvvdp(display_name=self._photometry, display_geometry=self._geom, quiet=True,
                                device=self.device, gpu_mem=4)
        else:
            from . import flip_torch
            net = flip_torch
        if isinstance(net, torch.nn.Module):
            for p in net.parameters():
                p.requires_grad_(False)
        self._nets[key] = net
        return net
    @property
    def cvvdp(self):
        return self._net("cvvdp")
    def __call__(self, name, t, r):
        if name == "l1":
            return (t - r).abs().mean(dim=(1, 2, 3))
        if name == "mse":
            return ((t - r) ** 2).mean(dim=(1, 2, 3))
        if name == "psnr":
            return -10 * torch.log10(((t - r) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12))
        if name == "ssim":
            return nvsqa_metrics.ssim(t, r)
        if name == "ms_ssim":
            from pytorch_msssim import ms_ssim
            return ms_ssim(t, r, data_range=1.0, size_average=False)
        if name in ("lpips_vgg", "lpips_alex"):
            return self._net(name)(t, r, normalize=True).reshape(-1)
        if name == "dists":
            return self._net("dists")(t, r, require_grad=True, batch_average=False).reshape(-1)
        if name == "flip":
            ft = self._net("flip")
            tc, rc = t.clamp(0, 1), r.clamp(0, 1)
            err = ft.compute_ldrflip(ft.color_space_transform(tc, "srgb2ycxcz"),
                                     ft.color_space_transform(rc, "srgb2ycxcz"),
                                     self.ppd, 0.7, 0.5, 0.4, 0.95, 1e-15)
            return err.mean(dim=(1, 2, 3))
        if name == "cvvdp_image":
            m = self.cvvdp
            out = []
            for i in range(t.shape[0]):
                jod, _ = m.predict(t[i:i + 1].permute(0, 2, 3, 1).contiguous(),
                                   r[i:i + 1].permute(0, 2, 3, 1).contiguous(), dim_order="BHWC")
                out.append(jod.reshape(-1))
            return torch.cat(out)
        if name == "dreamsim":
            def prep(x):
                return F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            return self._net("dreamsim")(prep(t), prep(r)).reshape(-1)
        if name == "fsim":
            import piq
            return piq.fsim(t.clamp(0, 1), r.clamp(0, 1), data_range=1.0, reduction="none", chromatic=True)
        if name == "vif":
            import piq
            return piq.vif_p(t.clamp(0, 1), r.clamp(0, 1), data_range=1.0, reduction="none")
    def loss(self, name, t, r):
        return loss_form(LOSSES[name], self(LOSSES[name], t, r)).mean()
    @torch.no_grad()
    def evaluate(self, test_u8, ref_u8, names):
        test_u8 = torch.as_tensor(np.asarray(test_u8))
        ref_u8 = torch.as_tensor(np.asarray(ref_u8))
        out = {n: [] for n in names}
        for i in range(test_u8.shape[0]):
            t = test_u8[i:i + 1].to(self.device).permute(0, 3, 1, 2).float() / 255.0
            r = ref_u8[i:i + 1].to(self.device).permute(0, 3, 1, 2).float() / 255.0
            for n in names:
                out[n].append(float(self(n, t, r)[0]))
        return {n: np.asarray(v) for n, v in out.items()}
    @torch.no_grad()
    def video(self, test_u8, ref_u8, fps):
        m = self.cvvdp
        jod, stats = m.predict(torch.as_tensor(test_u8).to(self.device), torch.as_tensor(ref_u8).to(self.device),
                               dim_order="FHWC", frames_per_second=float(fps))
        q = torch.as_tensor(stats["Q_per_ch"], device=self.device)
        res = {"cvvdp": float(jod), "cvvdp_sust": float(m.do_pooling_and_jods(q[:, :3]))}
        del q, stats
        torch.cuda.empty_cache()
        return res
