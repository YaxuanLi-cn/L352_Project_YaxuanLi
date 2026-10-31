import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _luma(x):
    x = x.float() / 255.0
    return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[:, None]


def _grad_mag(y):
    kx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=y.device).view(1, 1, 3, 3) / 8
    gx = F.conv2d(y, kx)
    gy = F.conv2d(y, kx.transpose(2, 3))
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
@torch.no_grad()


def describe(test, ref, device="cuda", flow_every=10, chunk=32):
    test_t, ref_t = torch.as_tensor(test), torch.as_tensor(ref)
    n = test_t.shape[0]
    acc = {k: [] for k in ("sharp_ratio", "err_top1", "bias", "mse")}
    dt_test, dt_ref, dt_err = [], [], []
    prev = None
    for i in range(0, n, chunk):
        yt = _luma(test_t[i:i + chunk].to(device))
        yr = _luma(ref_t[i:i + chunk].to(device))
        err = yt - yr
        acc["sharp_ratio"].append(_grad_mag(yt).mean(dim=(1, 2, 3)) / _grad_mag(yr).mean(dim=(1, 2, 3)))
        e2 = (err ** 2).flatten(1)
        k = max(1, e2.shape[1] // 100)
        acc["err_top1"].append(torch.topk(e2, k, dim=1).values.sum(1) / e2.sum(1).clamp_min(1e-12))
        acc["bias"].append(err.mean(dim=(1, 2, 3)))
        acc["mse"].append(e2.mean(1))
        if prev is not None:
            yt_all, yr_all = torch.cat([prev[0], yt]), torch.cat([prev[1], yr])
        else:
            yt_all, yr_all = yt, yr
        if yt_all.shape[0] > 1:
            d_t = yt_all[1:] - yt_all[:-1]
            d_r = yr_all[1:] - yr_all[:-1]
            dt_test.append((d_t ** 2).mean(dim=(1, 2, 3)))
            dt_ref.append((d_r ** 2).mean(dim=(1, 2, 3)))
            dt_err.append(((d_t - d_r) ** 2).mean(dim=(1, 2, 3)))
        prev = (yt[-1:], yr[-1:])
    out = {k: float(torch.cat(v).mean()) for k, v in acc.items()}
    if dt_test:
        dtt, dtr, dte = (float(torch.cat(v).mean()) for v in (dt_test, dt_ref, dt_err))
        out["temp_ratio"] = dtt / max(dtr, 1e-12)
        out["temp_err"] = dte / max(2 * out["mse"], 1e-12)
    tn, rn = test_t.numpy(), ref_t.numpy()
    mags = []
    for i in range(0, n, flow_every):
        a = cv2.cvtColor(rn[i], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(tn[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 7, 1.5, 0)
        mags.append(np.median(np.linalg.norm(flow, axis=-1)))
    out["shift_px"] = float(np.mean(mags))
    return out
