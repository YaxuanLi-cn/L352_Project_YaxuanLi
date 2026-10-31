import glob
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import paths as P              
from gh import metrics as M              
DEV = "cuda"
OPTIMUM = {"l1": 0.0, "mse": 0.0, "ssim": 1.0, "ms_ssim": 1.0, "lpips_vgg": 0.0, "lpips_alex": 0.0,
           "dists": 0.0, "flip": 0.0, "cvvdp_image": 10.0, "dreamsim": 0.0, "fsim": 1.0, "vif": 1.0}
_SUITE = None
RESULTS = {}


def suite():
    global _SUITE
    if _SUITE is None:
        _SUITE = M.Metrics(DEV)
    return _SUITE


def images(n=2, size=384):
    import cv2
    files = sorted(glob.glob(os.path.join(P.DATASETS, "mipnerf360", "garden", "images_4", "*")))[:n]
    ims = []
    for f in files:
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)[100:100 + size, 200:200 + size]
        ims.append(im)
    ref = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().div(255).to(DEV)
    g = torch.Generator(device="cpu").manual_seed(0)
    blur = F.avg_pool2d(F.pad(ref, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    noise = 0.03 * torch.randn(ref.shape, generator=g).to(DEV)
    test = (blur + noise + torch.tensor([0.02, 0.0, -0.02], device=DEV).view(1, 3, 1, 1)).clamp(0, 1)
    return (test * 255).round() / 255, ref


def test_optimum_and_gradient():
    m = suite()
    t, r = images()
    for name in M.IMAGE_METRICS:
        if name == "psnr":
            continue
        same = m(name, r, r)
        diff = m(name, t, r)
        x = t.clone().requires_grad_(True)
        lf = M.loss_form(name, m(name, x, r)).mean()
        lf.backward()
        gn = float(x.grad.norm())
        RESULTS.setdefault(name, {}).update(identity=same.tolist(), distorted=diff.tolist(), grad_norm=gn,
                                            grad_finite=bool(torch.isfinite(x.grad).all()))
        assert np.allclose(same.detach().cpu().numpy(), OPTIMUM[name], atol=2e-3), (name, same)
        better = M.SIGN[name] * (same - diff)
        assert (better > 0).all(), (name, same, diff)
        assert torch.isfinite(x.grad).all() and gn > 0, (name, gn)
    x = t.clone().requires_grad_(True)
    (-m("psnr", x, r).mean()).backward()
    assert torch.isfinite(x.grad).all() and float(x.grad.norm()) > 0
    RESULTS.setdefault("psnr", {}).update(identity=m("psnr", r, r).tolist(), distorted=m("psnr", t, r).tolist(),
                                         grad_norm=float(x.grad.norm()))


def test_reference_implementations():
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    m = suite()
    t, r = images()
    tn = t.permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)
    rn = r.permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)
    tu8 = (tn * 255).round().astype(np.uint8)
    ru8 = (rn * 255).round().astype(np.uint8)
    rows = {}
    rows["psnr"] = ([peak_signal_noise_ratio(rn[i], tn[i], data_range=1.0) for i in range(len(tn))],
                    m("psnr", t, r).tolist(), 1e-3)
    rows["ssim"] = ([structural_similarity(rn[i], tn[i], data_range=1.0, channel_axis=2, gaussian_weights=True,
                                           sigma=1.5, use_sample_covariance=False) for i in range(len(tn))],
                    m("ssim", t, r).tolist(), 1e-3)
    import piq
    rows["ms_ssim"] = (piq.multi_scale_ssim(t, r, data_range=1.0, reduction="none").tolist(),
                       m("ms_ssim", t, r).tolist(), 5e-3)
    t2, r2 = t[..., :256, :256].contiguous(), r[..., :256, :256].contiguous()
    rows["dists"] = (piq.DISTS(reduction="none").to(DEV)(t2, r2).tolist(), m("dists", t2, r2).tolist(), 2e-3)
    import flip_evaluator
    ref_flip = [float(flip_evaluator.evaluate(ru8[i].astype(np.float32) / 255, tu8[i].astype(np.float32) / 255, "LDR",
                                              inputsRGB=True, applyMagma=False, computeMeanError=True,
                                              parameters={"ppd": m.ppd})[1]) for i in range(len(tn))]
    rows["flip"] = (ref_flip, m("flip", t, r).tolist(), 2e-3)
    cv = m.cvvdp
    jod_u8 = [float(cv.predict(torch.from_numpy(tu8[i:i + 1]).to(DEV), torch.from_numpy(ru8[i:i + 1]).to(DEV),
                               dim_order="BHWC")[0]) for i in range(len(tn))]
    rows["cvvdp_image"] = (jod_u8, m("cvvdp_image", t, r).tolist(), 1e-3)
    loss_ref = [float(cv.loss(t[i:i + 1].permute(0, 2, 3, 1), r[i:i + 1].permute(0, 2, 3, 1), dim_order="BHWC"))
                for i in range(len(tn))]
    rows["cvvdp_loss"] = (loss_ref, M.loss_form("cvvdp_image", m("cvvdp_image", t, r)).tolist(), 1e-3)
    from nvsqa.metrics import MetricSuite
    old = MetricSuite(metrics=["lpips_alex", "lpips_vgg", "psnr", "ssim", "ms_ssim"])
    o = old.images(torch.from_numpy(tu8), torch.from_numpy(ru8))
    for k in ("lpips_alex", "lpips_vgg", "psnr", "ssim", "ms_ssim"):
        rows[f"{k}_vs_existing_project"] = (o[k].tolist(), m(k, t, r).tolist(), 1e-4)
    for k, (ref_v, ours, tol) in rows.items():
        err = float(np.max(np.abs(np.asarray(ref_v) - np.asarray(ours))))
        RESULTS.setdefault("reference_" + k, {}).update(reference=list(map(float, ref_v)), ours=list(map(float, ours)),
                                                         max_abs_diff=err, tolerance=tol, ok=err <= tol)
    bad = {k: v for k, v in RESULTS.items() if k.startswith("reference_") and not v["ok"]}
    assert not bad, bad
if __name__ == "__main__":
    failures = []
    for fn in (test_optimum_and_gradient, test_reference_implementations):
        try:
            fn()
            print(fn.__name__, "OK")
        except AssertionError as e:
            failures.append(f"{fn.__name__}: {e}")
            print(fn.__name__, "FAILED", e)
    out = os.path.join(P.ORIG_RESULTS, "unit_tests.json")
    with open(out, "w") as f:
        json.dump(dict(results=RESULTS, failures=failures), f, indent=1)
    for k, v in RESULTS.items():
        print(k, {a: (np.round(b, 5).tolist() if isinstance(b, (list, float)) else b) for a, b in v.items()})
    sys.exit(1 if failures else 0)
