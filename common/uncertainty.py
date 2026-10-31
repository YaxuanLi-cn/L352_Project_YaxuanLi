import numpy as np
LUMA = np.array([0.2126, 0.7152, 0.0722])


def to_float(x):
    x = np.asarray(x)
    return x.astype(np.float64) / 255.0 if x.dtype == np.uint8 else x.astype(np.float64)


def variance_map(renders):
    x = to_float(renders)
    return x.var(axis=0, ddof=1).mean(axis=-1)


def bias2_map(renders, ref, V=None):
    x = to_float(renders)
    r = to_float(ref)
    k = x.shape[0]
    V = variance_map(x) if V is None else V
    b2 = ((x.mean(axis=0) - r) ** 2).mean(axis=-1) - V / k
    return np.maximum(b2, 0.0)


def texture_map(ref):
    import cv2
    y = to_float(ref) @ LUMA
    gx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def tile_grid(h, w, t):
    rows, cols = h // t, w // t
    return (h - rows * t) // 2, (w - cols * t) // 2, rows, cols


def tile_view(a, t):
    y0, x0, rows, cols = tile_grid(a.shape[0], a.shape[1], t)
    c = a[y0:y0 + rows * t, x0:x0 + cols * t]
    c = c.reshape(rows, t, cols, t, *a.shape[2:])
    c = np.moveaxis(c, 2, 1)
    return c.reshape(rows * cols, t, t, *a.shape[2:])


def tile_stats(V, B2, G, t):
    return dict(U=np.sqrt(tile_view(V, t).mean(axis=(1, 2))), B=np.sqrt(tile_view(B2, t).mean(axis=(1, 2))),
                G=tile_view(G, t).mean(axis=(1, 2)))


def assign_strata(u, q=5):
    u = np.asarray(u, dtype=np.float64)
    n = len(u)
    order = np.argsort(u, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    stratum = (rank * q) // n + 1
    edges = np.array([[u[stratum == k].min(), u[stratum == k].max()] for k in range(1, q + 1)])
    return stratum, edges


def self_score(z, own, others):
    return z[..., own] - z[..., others].mean(axis=-1)


def stratum_means(values, stratum, q=5):
    out = np.stack([values[stratum == k].mean(axis=0) for k in range(1, q + 1)])
    sizes = np.array([(stratum == k).sum() for k in range(1, q + 1)])
    return out, sizes
