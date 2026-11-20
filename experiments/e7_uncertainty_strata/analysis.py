#!/usr/bin/env python
import argparse
import itertools
import json
import os
import sys
import warnings
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import helpers as C              
import numpy as np              
import pandas as pd              
from scipy.stats import kendalltau              
from gh import metrics as M              
IMG = list(M.IMAGE_METRICS)
COMMON7 = ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg", "flip", "cvvdp_image"]
B = 2000
Q = 5
CTRL = "default__ctrl"
DEFAULT_FAMILIES = {"pixel", "ssim"}
STRATA = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def loss_of(model):
    return model.split("__")[0]


def group_of(model):
    return model.split("__")[1]


def own_metric(model):
    return M.LOSSES[loss_of(model)]


def reference_sets(metrics, j, mode):
    fam = M.FAMILY[j]
    if mode == "nondefault":
        return [[i for i, m in enumerate(metrics) if M.FAMILY[m] == f]
                for f in sorted({M.FAMILY[m] for m in metrics}) if f != fam and f not in DEFAULT_FAMILIES]
    groups = [[i for i, m in enumerate(metrics) if M.FAMILY[m] == f] for f in sorted({M.FAMILY[m] for m in metrics}) if f != fam]
    return [sum(groups, [])] if mode == "paper" else groups


def ref_z(z, metrics, j, mode):
    sets = reference_sets(metrics, j, mode)
    sets = [x for x in sets if x]
    if not sets:
        return np.full(z.shape[:-1], np.nan)
    if len(sets) == 1:
        return z[..., sets[0]].mean(axis=-1)
    return np.mean([z[..., s].mean(axis=-1) for s in sets], axis=0)


def boot_scenes(per_scene, rng, stat=np.mean, n=B):
    per_scene = np.asarray(per_scene, dtype=float)
    k = per_scene.shape[0]
    idx = rng.integers(0, k, (n, k))
    draws = np.stack([stat(per_scene[i], axis=0) for i in idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return stat(per_scene, axis=0), np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0), draws


class Data:
    def __init__(self, out, drop=(), lt=None):
        self.out = out
        lt = pd.read_parquet(os.path.join(out, "tiles_long.parquet")) if lt is None else lt
        self.lt = lt
        self.scenes = sorted(lt.scene.unique())
        self.models = [CTRL] + sorted(m for m in lt.model.unique() if m != CTRL)
        self.metrics = [m for m in IMG if m in set(lt.metric) and m not in set(drop)]
        self.loss_models = [m for m in self.models if m != CTRL and own_metric(m) in self.metrics]
        self._arr = {}
        self.wide = {}
        for (sc, sp, sd), g in lt.groupby(["scene", "split", "seed"]):
            w = g.pivot_table(index=["view", "tile"], columns=["model", "metric"], values="value")
            meta = g.drop_duplicates(["view", "tile"]).set_index(["view", "tile"])[["U", "B", "G", "quintile"]]
            self.wide[(sc, sp, sd)] = (w.sort_index(), meta.loc[w.sort_index().index])
    def seeds(self, scene, split):
        return sorted(sd for (sc, sp, sd) in self.wide if sc == scene and sp == split)
    def tile_values(self, scene, split, seed, model):
        key = (scene, split, seed, model)
        if key not in self._arr:
            w, _ = self.wide[(scene, split, seed)]
            self._arr[key] = w[model][self.metrics].to_numpy()
        return self._arr[key]
    def meta(self, scene, split, seed=0):
        return self.wide[(scene, split, seed)][1]


def stratum_scores(d, scene, split, seed, quintile, extra_groups=True):
    out = {}
    for m in d.models:
        x = d.tile_values(scene, split, seed, m)
        rows = [x[quintile == k].mean(axis=0) for k in range(1, Q + 1)] + [x.mean(axis=0)]
        if extra_groups:
            rows.append(x[(quintile == 1) | (quintile == 2)].mean(axis=0))
        out[m] = np.stack(rows)
    return out


def improvements(d, quint_of=None):
    res = {}
    for sc in d.scenes:
        for sp in ("test", "train"):
            seeds = d.seeds(sc, sp)
            if not seeds:
                continue
            per_seed = []
            for sd in seeds:
                q = d.meta(sc, sp, sd).quintile.to_numpy() if quint_of is None else quint_of(sc, sp, sd)
                s = stratum_scores(d, sc, sp, sd, q)
                per_seed.append({m: s[m] - s[CTRL] for m in d.loss_models})
            res[(sc, sp)] = {m: np.stack([p[m] for p in per_seed]) for m in d.loss_models}
    return res


def column_scale(d, delta):
    rows = [delta[(sc, "test")][m][0, Q] for sc in d.scenes for m in d.loss_models]
    s = np.std(np.stack(rows), axis=0, ddof=1)
    s[s == 0] = np.nan
    return s


def self_terms(d, z, model, mode):
    j = d.metrics.index(own_metric(model))
    own = z[..., j]
    ref = ref_z(z, d.metrics, own_metric(model), mode)
    return own, ref, own - ref


def per_scene_group(d, delta, scale, split, group, mode="paper", seed_avg=True):
    O, R, S = [], [], []
    for sc in d.scenes:
        if (sc, split) not in delta:
            continue                               
        o, r, s = [], [], []
        for m in d.loss_models:
            if group_of(m) != group:
                continue
            z = delta[(sc, split)][m] / scale
            a, b, c = self_terms(d, z, m, mode)
            if seed_avg:
                a, b, c = a.mean(0), b.mean(0), c.mean(0)
            o.append(a), r.append(b), s.append(c)
        O.append(np.mean(o, axis=0)), R.append(np.mean(r, axis=0)), S.append(np.mean(s, axis=0))
    return np.stack(O), np.stack(R), np.stack(S)


def sums_own_ref(d, delta, scale, split, group):
    O, R = [], []
    for sc in d.scenes:
        o = r = 0.0
        for m in d.loss_models:
            if group_of(m) != group:
                continue
            a, b, _ = self_terms(d, (delta[(sc, split)][m] / scale).mean(0), m, "paper")
            o, r = o + a, r + b
        O.append(o), R.append(r)
    return np.stack(O), np.stack(R)


def a1_a2(d, delta, scale, rng, tag=""):
    rows, crows, verdict = [], [], {}
    for split in ("test", "train"):
        for group in ("pure", "mixed"):
            for mode in ("paper", "families", "nondefault"):
                O, R, S = per_scene_group(d, delta, scale, split, group, mode)
                for gi, gname in enumerate(STRATA + ["overall"]):
                    for name, arr in (("own_z", O), ("ref_z", R), ("S", S)):
                        p, lo, hi, _ = boot_scenes(arr[:, gi], rng)
                        rows.append(dict(split=split, group=group, reference=mode, stratum=gname, quantity=name, value=p,
                                         lo=lo, hi=hi, n_scenes=arr.shape[0]))
                if mode == "paper":
                    diff = S[:, Q - 1] - S[:, 0]
                    p, lo, hi, _ = boot_scenes(diff, rng)
                    verdict[(split, group, "a")] = dict(diff=p, lo=lo, hi=hi, n_pos=int((diff > 0).sum()),
                                                        n=len(diff), per_scene=diff.tolist())
            so, sr = sums_own_ref(d, delta, scale, split, group)
            O, _, _ = per_scene_group(d, delta, scale, split, group, "paper")
            k = so.shape[0]
            idx = rng.integers(0, k, (B, k))
            for gi, gname in enumerate(STRATA + ["overall", "Q1+Q2"]):
                _, olo, ohi, _ = boot_scenes(O[:, gi], rng)
                c = sr[:, gi].sum() / so[:, gi].sum()
                draws = np.array([sr[i, gi].sum() / so[i, gi].sum() for i in idx])
                crows.append(dict(split=split, group=group, stratum=gname, C=c, lo=np.percentile(draws, 2.5),
                                  hi=np.percentile(draws, 97.5), own_lo=olo, own_hi=ohi, reported=bool(olo > 0)))
            low, high = STRATA.__len__() + 1, Q - 1                           
            dd = np.array([sr[i, low].sum() / so[i, low].sum() - sr[i, high].sum() / so[i, high].sum() for i in idx])
            _, olo_low, _, _ = boot_scenes(O[:, low], rng)
            verdict[(split, group, "b")] = dict(
                C_low=sr[:, low].sum() / so[:, low].sum(), C_high=sr[:, high].sum() / so[:, high].sum(),
                diff=sr[:, low].sum() / so[:, low].sum() - sr[:, high].sum() / so[:, high].sum(),
                lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5), own_low_lo=olo_low)
    a1 = pd.DataFrame(rows)
    a2 = pd.DataFrame(crows)
    return a1, a2, verdict


def per_scene_table(d, delta, scale):
    rows = []
    for split in ("test", "train"):
        for group in ("pure", "mixed"):
            O, R, S = per_scene_group(d, delta, scale, split, group, "paper")
            scs = [sc for sc in d.scenes if (sc, split) in delta]
            for i, sc in enumerate(scs):
                for gi, gname in enumerate(STRATA + ["overall"]):
                    rows.append(dict(scene=sc, split=split, group=group, stratum=gname, own_z=O[i, gi], ref_z=R[i, gi],
                                     S=S[i, gi]))
    return pd.DataFrame(rows)


def seed_sd(d, delta, scale):
    rows = []
    for split in ("test", "train"):
        for sc in d.scenes:
            if (sc, split) not in delta or delta[(sc, split)][d.loss_models[0]].shape[0] < 2:
                continue
            for group in ("pure", "mixed"):
                per = []
                for m in d.loss_models:
                    if group_of(m) == group:
                        _, _, s = self_terms(d, delta[(sc, split)][m] / scale, m, "paper")
                        per.append(s)
                per = np.mean(per, axis=0)
                for gi, gname in enumerate(STRATA + ["overall"]):
                    rows.append(dict(scene=sc, split=split, group=group, stratum=gname,
                                     **{f"S_seed{k}": per[k, gi] for k in range(per.shape[0])},
                                     sd_over_seeds=np.std(per[:, gi], ddof=1)))
    return pd.DataFrame(rows)


def a3_concentration(d, scale, rng):
    curves, shares = [], []
    grid = np.linspace(0, 1, 101)
    for split in ("test", "train"):
        for group in ("pure", "mixed"):
            for sc in d.scenes:
                seeds = d.seeds(sc, split)
                if not seeds:
                    continue
                meta = d.meta(sc, split, seeds[0])
                s_tile = []
                for sd in seeds:
                    ctrl = d.tile_values(sc, split, sd, CTRL)
                    per_m = []
                    for m in d.loss_models:
                        if group_of(m) == group:
                            z = (d.tile_values(sc, split, sd, m) - ctrl) / scale
                            per_m.append(self_terms(d, z, m, "paper")[2])
                    s_tile.append(np.mean(per_m, axis=0))
                s_tile = np.mean(s_tile, axis=0)
                mse_ctrl = -d.tile_values(sc, split, seeds[0], CTRL)[:, d.metrics.index("mse")]
                order = np.argsort(-meta.U.to_numpy(), kind="stable")
                n = len(order)
                frac = np.arange(1, n + 1) / n
                cs = np.cumsum(s_tile[order]) / s_tile.sum()
                ce = np.cumsum(mse_ctrl[order]) / mse_ctrl.sum()
                curves.append(dict(split=split, group=group, scene=sc, x=grid.tolist(),
                                   S=np.interp(grid, frac, cs).tolist(), err=np.interp(grid, frac, ce).tolist()))
                top = order[: int(round(0.2 * n))]
                shares.append(dict(split=split, group=group, scene=sc, total_S=float(s_tile.sum()),
                                   share_S_top20=float(s_tile[top].sum() / s_tile.sum()),
                                   share_err_top20=float(mse_ctrl[top].sum() / mse_ctrl.sum())))
    sh = pd.DataFrame(shares)
    summ = []
    for (split, group), g in sh.groupby(["split", "group"]):
        for col in ("share_S_top20", "share_err_top20"):
            p, lo, hi, _ = boot_scenes(g[col].to_numpy(), rng)
            summ.append(dict(split=split, group=group, quantity=col, value=p, lo=lo, hi=hi, n_scenes=len(g)))
    return pd.DataFrame(curves), sh, pd.DataFrame(summ)


def a5_regression(d, scale, rng):
    rows = []
    for split in ("test", "train"):
        for group in ("pure", "mixed"):
            data = []
            for sc in d.scenes:
                seeds = d.seeds(sc, split)
                if not seeds:
                    continue
                meta = d.meta(sc, split, seeds[0])
                s_tile = []
                for sd in seeds:
                    ctrl = d.tile_values(sc, split, sd, CTRL)
                    s_tile.append(np.mean([self_terms(d, (d.tile_values(sc, split, sd, m) - ctrl) / scale, m, "paper")[2]
                                           for m in d.loss_models if group_of(m) == group], axis=0))
                df = pd.DataFrame(dict(S=np.mean(s_tile, axis=0), U=meta.U.to_numpy(), B=meta.B.to_numpy(),
                                       G=meta.G.to_numpy()))
                for c in ("S", "U", "B", "G"):                         
                    r = df[c].rank().to_numpy()
                    df[c] = (r - r.mean()) / r.std() if r.std() > 0 else 0.0
                df["scene"] = sc
                data.append(df)
            def fit(frames):
                X = pd.concat(frames, ignore_index=True)
                dummies = pd.get_dummies(X.scene, drop_first=False).to_numpy(float)
                A = np.column_stack([X[["U", "B", "G"]].to_numpy(), dummies])
                coef, *_ = np.linalg.lstsq(A, X.S.to_numpy(), rcond=None)
                return coef[:3]
            point = fit(data)
            k = len(data)
            draws = []
            for i in rng.integers(0, k, (B, k)):
                fr = []
                for j, ii in enumerate(i):                       
                    f = data[ii].copy()
                    f["scene"] = f"{f.scene.iloc[0]}#{j}"
                    fr.append(f)
                draws.append(fit(fr))
            draws = np.array(draws)
            for i, v in enumerate(("U", "B", "G")):
                rows.append(dict(split=split, group=group, variable=v, coef=point[i], lo=np.percentile(draws[:, i], 2.5),
                                 hi=np.percentile(draws[:, i], 97.5), n_tiles=sum(len(x) for x in data), n_scenes=k))
    return pd.DataFrame(rows)


def kendall_mean(X, metrics):
    vals = []
    for a, b in itertools.combinations(range(len(metrics)), 2):
        vals.append(kendalltau(X[:, a], X[:, b]).statistic)
    return float(np.nanmean(vals))


def concordance(X, thr=None):
    c = dd = 0
    n_m = X.shape[1]
    for i, j in itertools.combinations(range(X.shape[0]), 2):
        diff = X[i] - X[j]
        for a, b in itertools.combinations(range(n_m), 2):
            if diff[a] == 0 or diff[b] == 0:
                continue
            if thr is not None and not (abs(diff[a]) > thr[a] and abs(diff[b]) > thr[b]):
                continue
            if diff[a] * diff[b] > 0:
                c += 1
            else:
                dd += 1
    return ((c - dd) / (c + dd) if c + dd else np.nan), c + dd


def a4_agreement(d, out, rng):
    import glob
    cm = [m for m in COMMON7 if m in d.metrics]
    ci = [d.metrics.index(m) for m in cm]
    thr_loss = {}
    for sc in d.scenes:
        if len(d.seeds(sc, "test")) < 2:
            continue
        sc_by_seed = {sd: stratum_scores(d, sc, "test", sd, d.meta(sc, "test", sd).quintile.to_numpy(), False)
                      for sd in d.seeds(sc, "test")}
        for gi in range(Q + 1):
            diffs = [np.abs(sc_by_seed[a][m][gi, ci] - sc_by_seed[b][m][gi, ci])
                     for m in d.models for a, b in itertools.combinations(sorted(sc_by_seed), 2)]
            thr_loss[gi] = np.percentile(np.stack(diffs), 90, axis=0)
    lad = {}
    for f in glob.glob(os.path.join(out, "cache_ladder", "*", "*.parquet")):
        sc = os.path.basename(os.path.dirname(f)).replace("_", "/", 1)
        g = pd.read_parquet(f)
        g = g.pivot_table(index=["view", "tile"], columns="metric", values="value").sort_index()
        lad[(sc, os.path.basename(f)[:-8])] = g
    thr_lad = {}
    rows = []
    for gi, gname in enumerate(STRATA + ["overall"]):
        diffs = []
        for (sc, cond), g in lad.items():
            if cond.endswith("-s1") and (sc, cond[:-3]) in lad:
                q = d.meta(sc, "test").quintile.loc[g.index].to_numpy()
                sel = (q == gi + 1) if gi < Q else np.ones(len(q), bool)
                diffs.append(np.abs(g[cm].to_numpy()[sel].mean(0) - lad[(sc, cond[:-3])][cm].to_numpy()[sel].mean(0)))
        thr_lad[gi] = np.percentile(np.stack(diffs), 90, axis=0) if diffs else None
        for source in ("loss", "ladder"):
            per_scene = []
            for sc in d.scenes:
                if source == "loss":
                    q = d.meta(sc, "test", 0).quintile.to_numpy()
                    s = stratum_scores(d, sc, "test", 0, q, False)
                    X = np.stack([s[m][gi, ci] for m in d.models])
                    thr = thr_loss.get(gi)
                else:
                    conds = sorted(c for (s2, c) in lad if s2 == sc and not c.endswith("-s1"))
                    if len(conds) < 3:
                        continue
                    q = d.meta(sc, "test").quintile.loc[lad[(sc, conds[0])].index].to_numpy()
                    sel = (q == gi + 1) if gi < Q else np.ones(len(q), bool)
                    X = np.stack([lad[(sc, c)][cm].to_numpy()[sel].mean(0) for c in conds])
                    thr = thr_lad.get(gi)
                tau = kendall_mean(X, cm)
                c_all, n_all = concordance(X)
                c_res, n_res = concordance(X, thr) if thr is not None else (np.nan, 0)
                per_scene.append((sc, tau, c_all, c_res, n_all, n_res))
            if not per_scene:
                continue
            arr = np.array([p[1:4] for p in per_scene], float)
            for qi, qname in enumerate(("kendall_tau", "concordance_all_pairs", "concordance_resolved_pairs")):
                ok = np.isfinite(arr[:, qi])
                if not ok.any():
                    continue
                p, lo, hi, _ = boot_scenes(arr[ok, qi], rng)
                rows.append(dict(source=source, stratum=gname, quantity=qname, value=p, lo=lo, hi=hi,
                                 n_scenes=int(ok.sum()), n_pairs=int(sum(x[4 if qi < 2 else 5] for x in per_scene))))
    thr_tab = pd.DataFrame([dict(source=s, stratum=(STRATA + ["overall"])[gi], **dict(zip(cm, v)))
                            for s, t in (("loss", thr_loss), ("ladder", thr_lad)) for gi, v in t.items() if v is not None])
    return pd.DataFrame(rows), thr_tab


def shuffled_strata(d, perm_rng):
    import uncertainty as UU
    out = {}
    for (sc, sp, sd), (w, meta) in d.wide.items():
        if (sc, sp) in out:
            continue                         
        u = meta.U.to_numpy().copy()
        views = meta.index.get_level_values(0).to_numpy()
        for v in np.unique(views):
            idx = np.where(views == v)[0]
            u[idx] = u[perm_rng.permutation(idx)]
        out[(sc, sp)] = UU.assign_strata(u, Q)[0]
    return out


def negative_control(d, scale, rng, n_perm=200):
    shuffled = shuffled_strata(d, np.random.default_rng(12345))
    delta = improvements(d, quint_of=lambda sc, sp, sd: shuffled[(sc, sp)])
    rows = []
    for split in ("test", "train"):
        for group in ("pure", "mixed"):
            _, _, S = per_scene_group(d, delta, scale, split, group, "paper")
            for gi, gname in enumerate(STRATA + ["overall"]):
                p, lo, hi, _ = boot_scenes(S[:, gi], rng)
                rows.append(dict(split=split, group=group, stratum=gname, S=p, lo=lo, hi=hi))
            p, lo, hi, _ = boot_scenes(S[:, Q - 1] - S[:, 0], rng)
            rows.append(dict(split=split, group=group, stratum="Q5-Q1", S=p, lo=lo, hi=hi))
    obs_delta = improvements(d)
    perm_rng = np.random.default_rng(54321)
    null = {(sp, g): [] for sp in ("test", "train") for g in ("pure", "mixed")}
    for _ in range(n_perm):
        sh = shuffled_strata(d, perm_rng)
        dl = improvements(d, quint_of=lambda sc, sp, sd, sh=sh: sh[(sc, sp)])
        for sp, g in null:
            _, _, S = per_scene_group(d, dl, scale, sp, g, "paper")
            null[(sp, g)].append(float(np.mean(S[:, Q - 1] - S[:, 0])))
    prow = []
    for (sp, g), vals in null.items():
        _, _, S = per_scene_group(d, obs_delta, scale, sp, g, "paper")
        obs = float(np.mean(S[:, Q - 1] - S[:, 0]))
        v = np.asarray(vals)
        prow.append(dict(split=sp, group=g, observed_Q5_minus_Q1=obs, null_mean=float(v.mean()),
                         null_lo=float(np.percentile(v, 2.5)), null_hi=float(np.percentile(v, 97.5)),                    
                         share_null_ge_observed=float((v >= obs).mean()), n_perm=n_perm))
    return pd.DataFrame(rows), pd.DataFrame(prow)


def check_whole_image(d, out, runs_root):
    from scipy.stats import spearmanr
    import glob
    rows = []
    for split in ("test", "train"):
        tile_mean, whole = {}, {}
        for sc in d.scenes:
            meta_views = sorted(d.meta(sc, split, 0).index.get_level_values(0).unique())
            for m in d.models:
                key = (sc, m)
                tile_mean[key] = dict(zip(d.metrics, d.tile_values(sc, split, 0, m).mean(axis=0)))
                if split == "test":
                    f = os.path.join(runs_root, C.scene_dir(sc), "s0", m, "metrics.csv")
                    if not os.path.exists(f):
                        continue
                    g = pd.read_csv(f)
                    g = g[(g.split == "test") & g.view_id.isin(meta_views)]
                    whole[key] = {k: M.SIGN[k] * v for k, v in g.groupby("metric").value.mean().items() if k in M.SIGN}
                else:
                    f = os.path.join(out, "cache", C.scene_dir(sc), "train", "s0", f"{m}.parquet")
                    if not os.path.exists(f):
                        continue
                    g = pd.read_parquet(f)
                    g = g[g.tile == -1]
                    whole[key] = g.groupby("metric").value.mean().to_dict()                     
        for met in d.metrics:
            ks = [k for k in tile_mean if k in whole and met in whole[k]]
            if len(ks) < 3:
                continue
            a = [tile_mean[k][met] for k in ks]
            b = [whole[k][met] for k in ks]
            rho = spearmanr(a, b).statistic
            within = []
            for sc in d.scenes:
                kk = [k for k in ks if k[0] == sc]
                if len(kk) >= 5:
                    within.append(spearmanr([tile_mean[k][met] for k in kk], [whole[k][met] for k in kk]).statistic)
            rows.append(dict(split=split, metric=met, spearman=rho, n=len(ks), ok=bool(rho >= 0.9),
                             spearman_within_scene=float(np.nanmean(within)) if within else np.nan,
                             spearman_within_scene_min=float(np.nanmin(within)) if within else np.nan))
    return pd.DataFrame(rows)


def pixel_version(d, out, rng):
    import glob
    from tiles import PIXEL_METRICS
    files = glob.glob(os.path.join(out, "pixel", "*.parquet"))
    if not files:
        return None
    px = pd.concat([pd.read_parquet(f) for f in files])
    mets = [m for m in IMG if m in PIXEL_METRICS and m in d.metrics]
    models = [m for m in d.loss_models if own_metric(m) in mets]
    rows = []
    sc_ = sorted(px.scene.unique())
    for version in ("pixel", "tile"):
        delta = {}
        for sc in sc_:
            for sp in ("test", "train"):
                if version == "pixel":
                    g = px[(px.scene == sc) & (px.split == sp)]
                    if g.empty:
                        continue
                    seeds = sorted(g.seed.unique())
                    w = {sd: g[g.seed == sd].pivot_table(index="quintile", columns=["model", "metric"], values="value")
                         for sd in seeds}
                    n = g[(g.seed == seeds[0]) & (g.model == CTRL) & (g.metric == mets[0])].sort_values("quintile").n_pixels.to_numpy()
                    sc_vals = {sd: {m: np.vstack([w[sd][m][mets].to_numpy(),
                                                  (w[sd][m][mets].to_numpy() * n[:, None]).sum(0) / n.sum()])
                                    for m in [CTRL] + models} for sd in seeds}
                else:
                    seeds = d.seeds(sc, sp)
                    sc_vals = {}
                    for sd in seeds:
                        q = d.meta(sc, sp, sd).quintile.to_numpy()
                        s = stratum_scores(d, sc, sp, sd, q, False)
                        sc_vals[sd] = {m: s[m][:, [d.metrics.index(k) for k in mets]] for m in [CTRL] + models}
                delta[(sc, sp)] = {m: np.stack([sc_vals[sd][m] - sc_vals[sd][CTRL] for sd in seeds]) for m in models}
        scale = np.std(np.stack([delta[(sc, "test")][m][0, Q] for sc in sc_ for m in models]), axis=0, ddof=1)
        for split in ("test", "train"):
            for group in ("pure", "mixed"):
                S = []
                for sc in sc_:
                    if (sc, split) not in delta:
                        continue
                    s = []
                    for m in models:
                        if group_of(m) != group:
                            continue
                        z = delta[(sc, split)][m] / scale
                        j = mets.index(own_metric(m))
                        ref = ref_z(z, mets, own_metric(m), "paper")
                        s.append((z[..., j] - ref).mean(0))
                    S.append(np.mean(s, axis=0))
                S = np.stack(S)
                for gi, gname in enumerate(STRATA + ["overall"]):
                    p, lo, hi, _ = boot_scenes(S[:, gi], rng)
                    rows.append(dict(version=version, split=split, group=group, stratum=gname, S=p, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def judge(verdict):
    out = {}
    for group in ("pure", "mixed"):
        a = verdict[("test", group, "a")]
        b = verdict[("test", group, "b")]                    
        ok_a = bool(a["lo"] > 0 and a["n_pos"] >= 7)
        if not b["own_low_lo"] > 0:
            ok_b = None                            
        else:
            ok_b = bool(b["lo"] > 0)                         
        if ok_a and ok_b:
            v = "supported"                    
        elif ok_a and ok_b is None:
            v = "supported (no measurable gain in the certain strata)"                         
        elif ok_a:
            v = "partly supported"                   
        else:
            v = "not supported"                  
        out[group] = dict(a=a, b=b, criterion_a=ok_a, criterion_b=ok_b, verdict=v)
    return out


def main():                                 
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=P.PILOT)
    ap.add_argument("--runs-root", default=P.RERUN_RUNS)
    args = ap.parse_args()
    res = os.path.join(args.out, "results")
    os.makedirs(res, exist_ok=True)
    rng = np.random.default_rng(0)
    d = Data(args.out)
    delta = improvements(d)
    scale = column_scale(d, delta)
    pd.DataFrame(dict(metric=d.metrics, s_m=scale)).to_csv(os.path.join(res, "column_scale_tiles.csv"), index=False)
    a1, a2, verdict = a1_a2(d, delta, scale, rng)
    a1.to_csv(os.path.join(res, "A1_strata_S.csv"), index=False)
    a2.to_csv(os.path.join(res, "A2_confirmed_share.csv"), index=False)
    per_scene_table(d, delta, scale).to_csv(os.path.join(res, "per_scene_strata.csv"), index=False)
    seed_sd(d, delta, scale).to_csv(os.path.join(res, "seed_sd_truck.csv"), index=False)
    curves, shares, summ = a3_concentration(d, scale, rng)
    curves.to_json(os.path.join(res, "A3_curves.json"), orient="records")
    shares.to_csv(os.path.join(res, "A3_top20_by_scene.csv"), index=False)
    summ.to_csv(os.path.join(res, "A3_top20.csv"), index=False)
    a4, thr = a4_agreement(d, args.out, rng)
    a4.to_csv(os.path.join(res, "A4_agreement.csv"), index=False)
    thr.to_csv(os.path.join(res, "A4_seed_noise_thresholds.csv"), index=False)
    a5_regression(d, scale, rng).to_csv(os.path.join(res, "A5_rank_regression.csv"), index=False)
    neg, perm = negative_control(d, scale, rng)
    neg.to_csv(os.path.join(res, "negative_control.csv"), index=False)
    perm.to_csv(os.path.join(res, "negative_control_permutations.csv"), index=False)
    chk = check_whole_image(d, args.out, args.runs_root)
    chk.to_csv(os.path.join(res, "check_tile_vs_whole.csv"), index=False)
    pv = pixel_version(d, args.out, rng)
    if pv is not None:
        pv.to_csv(os.path.join(res, "pixel_vs_tile.csv"), index=False)
    json.dump(judge(verdict), open(os.path.join(res, "verdict.json"), "w"), indent=1, default=float)
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in verdict.items()}, open(os.path.join(res, "verdict_all.json"), "w"),
              indent=1, default=float)                                
    failed = sorted(set(chk[~chk.ok].metric), key=IMG.index)
    if failed:
        d2 = Data(args.out, drop=failed, lt=d.lt)
        delta2 = improvements(d2)
        scale2 = column_scale(d2, delta2)
        a1b, a2b, verdict2 = a1_a2(d2, delta2, scale2, np.random.default_rng(0))
        a1b.to_csv(os.path.join(res, "A1_strata_S__without_failed_metrics.csv"), index=False)
        a2b.to_csv(os.path.join(res, "A2_confirmed_share__without_failed_metrics.csv"), index=False)
        json.dump(dict(dropped_metrics=failed, loss_models=d2.loss_models, **judge(verdict2)),
                  open(os.path.join(res, "verdict_without_failed_metrics.json"), "w"), indent=1, default=float)                  
        C.log(f"metrics below 0.9 in the tile-vs-whole check: {failed}; A1, A2 and the verdict also written without them")
    C.log(f"results written to {res}")
if __name__ == "__main__":
    main()
