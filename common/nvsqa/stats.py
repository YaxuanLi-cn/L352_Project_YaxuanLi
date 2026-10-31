import itertools
import warnings
import numpy as np
from scipy import optimize, stats
MIN_SLOPE = 0.2


def logistic4(x, b1, b2, b3, b4):
    return b2 + (b1 - b2) / (1.0 + np.exp(np.clip(-(x - b3) / b4, -60.0, 60.0)))


def fit_logistic(x, y, quick=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sx = np.std(x) + 1e-12
    xn = (x - np.mean(x)) / sx
    sign = 1.0 if np.corrcoef(xn, y)[0, 1] >= 0 else -1.0
    def model(v, b1, b2, b3, theta):
        return logistic4(v, b1, b2, b3, sign * (MIN_SLOPE + np.exp(np.clip(theta, -20.0, 5.0))))
    best = None
    for theta0 in ((0.0,) if quick else (-1.0, 0.0, 1.0)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = optimize.curve_fit(model, xn, y, p0=[np.max(y), np.min(y), 0.0, theta0],
                                          maxfev=400 if quick else 5000)
            err = np.mean((model(xn, *p) - y) ** 2)
            if np.isfinite(err) and (best is None or err < best[0]):
                best = (err, p)
        except (RuntimeError, ValueError):
            continue
    lin = np.polyfit(xn, y, 1)
    lin_err = np.mean((np.polyval(lin, xn) - y) ** 2)
    mx, sd = np.mean(x), sx
    if best is None or best[0] > lin_err:
        return lambda v: np.polyval(lin, (np.asarray(v, float) - mx) / sd)
    p = best[1]
    return lambda v: model((np.asarray(v, float) - mx) / sd, *p)


def agreement(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    f = fit_logistic(x, y)
    yp = f(x)
    return {
        "srocc": stats.spearmanr(x, y).statistic,
        "krocc": stats.kendalltau(x, y).statistic,
        "plcc": stats.pearsonr(yp, y).statistic,
        "rmse": float(np.sqrt(np.mean((yp - y) ** 2))),
    }


def _fast_srocc(x, y):
    return np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1]


def bootstrap_agreement(preds, y, n_boot=2000, seed=0, measures=("srocc", "krocc", "plcc", "rmse"),
                        groups=None, n_boot_fit=400):
    rng = np.random.default_rng(seed)
    names = list(preds)
    y = np.asarray(y, float)
    n = len(y)
    X = {k: np.asarray(v, float) for k, v in preds.items()}
    est = {k: agreement(X[k], y) for k in names}
    samples = {m: np.full((n_boot, len(names)), np.nan) for m in measures}
    if groups is not None:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in ug}
    for b in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
        yb = y[idx]
        if np.ptp(yb) == 0:
            continue
        for j, k in enumerate(names):
            xb = X[k][idx]
            if np.ptp(xb) == 0:
                continue
            if "srocc" in measures:
                samples["srocc"][b, j] = _fast_srocc(xb, yb)
            if "krocc" in measures:
                samples["krocc"][b, j] = stats.kendalltau(xb, yb).statistic
            if ("plcc" in measures or "rmse" in measures) and b < n_boot_fit:
                yp = fit_logistic(xb, yb, quick=True)(xb)
                if "plcc" in measures:
                    samples["plcc"][b, j] = np.corrcoef(yp, yb)[0, 1]
                if "rmse" in measures:
                    samples["rmse"][b, j] = np.sqrt(np.mean((yp - yb) ** 2))
    summary = {}
    for j, k in enumerate(names):
        summary[k] = {}
        for m in measures:
            lo, hi = np.nanpercentile(samples[m][:, j], [2.5, 97.5])
            summary[k][m] = (est[k][m], lo, hi)
    return summary, samples, names


def paired_difference(samples, names, measure="srocc"):
    s = samples[measure]
    out = {}
    for a, b in itertools.combinations(range(len(names)), 2):
        d = s[:, a] - s[:, b]
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out[(names[a], names[b])] = dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), p=float(min(p, 1.0)))
    return out


def within_group_srocc(x, y, groups):
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 3 and np.ptp(x[m]) > 0 and np.ptp(y[m]) > 0:
            out[g] = stats.spearmanr(x[m], y[m]).statistic
        else:
            out[g] = np.nan
    return out


def within_group_test(x, y, groups, n_perm=10000, seed=0):
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    rng = np.random.default_rng(seed)
    rx, ry = [], []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 3:
            continue
        a, b = stats.rankdata(x[m]), stats.rankdata(y[m])
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        rx.append((a - a.mean()) / (a.std() * len(a)))
        ry.append((b - b.mean()) / b.std())
    per_group = np.array([float(a @ b) for a, b in zip(rx, ry)])
    obs = per_group.mean()
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.mean([float(rng.permutation(a) @ b) for a, b in zip(rx, ry)])
    p = (1 + (null >= obs).sum()) / (n_perm + 1)
    boot = [per_group[rng.integers(0, len(per_group), len(per_group))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(obs), p=float(p), lo=float(lo), hi=float(hi), n_groups=len(per_group))


def pairwise_accuracy(x, y, groups, min_delta=0.0):
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    agree, total = 0, 0
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        for i, j in itertools.combinations(idx, 2):
            dy = y[i] - y[j]
            if abs(dy) <= min_delta:
                continue
            dx = x[i] - x[j]
            total += 1
            agree += (dx * dy > 0) + 0.5 * (dx == 0)
    return (agree / total if total else np.nan), total


def pairwise_disagreement(a, b, groups):
    a, b, groups = np.asarray(a, float), np.asarray(b, float), np.asarray(groups)
    dis, total = 0, 0
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        da = a[idx][:, None] - a[idx][None, :]
        db = b[idx][:, None] - b[idx][None, :]
        iu = np.triu_indices(len(idx), 1)
        s = np.sign(da[iu]) * np.sign(db[iu])
        dis += (s < 0).sum()
        total += (s != 0).sum()
    return (dis / total if total else np.nan), int(total)
