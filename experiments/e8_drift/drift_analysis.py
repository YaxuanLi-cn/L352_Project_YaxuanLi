#!/usr/bin/env python
import glob
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
sys.path.insert(0, P.code(P.E7))
import helpers as C              
import uncertainty as U              
import analysis as A              
import numpy as np              
import pandas as pd              
from scipy.stats import rankdata, spearmanr              
from gh import metrics as M              
IMG = list(M.IMAGE_METRICS)
OUT = P.DRIFT
RES = os.path.join(OUT, "results")
PILOT = P.PILOT
CTRL, BASE = "default__ctrl", "none__base"
B = 2000
PSNR_CAP = 60.0
CLASS = {"l1": "pixel", "l2": "pixel", "dssim": "structural", "msssim": "structural", "lpips_vgg": "perceptual",
         "lpips_alex": "perceptual", "dists": "perceptual", "flip": "perceptual", "cvvdp": "perceptual"}


def dist(metric, value):
    value = np.asarray(value, dtype=float)
    if metric == "psnr":
        return PSNR_CAP - np.minimum(value, PSNR_CAP)
    return np.asarray(M.loss_form(metric, value))


def own_metric(model):
    return M.LOSSES[model.split("__")[0]]


def others(j, with_psnr=False):
    return [IMG.index(k) for k in IMG if M.FAMILY[k] != M.FAMILY[j] and (with_psnr or k != "psnr")]


def paper_scale():
    return pd.read_csv(os.path.join(P.ORIG_TABLES, "column_scale.csv")).set_index("metric").scale[IMG].to_numpy()


def load_pairs():
    fs = sorted(glob.glob(os.path.join(OUT, "pairs", "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["dist"] = 0.0
    for m_, g in df.groupby("metric"):
        df.loc[g.index, "dist"] = dist(m_, g.value.to_numpy())
    return df


def cube(df, scenes, models, col="dist"):
    t = df.groupby(["scene", "model", "metric"])[col].mean()
    out = np.full((len(scenes), len(models), len(IMG)), np.nan)
    for si, s in enumerate(scenes):
        for ci, c in enumerate(models):
            for mi, m_ in enumerate(IMG):
                if (s, c, m_) in t.index:
                    out[si, ci, mi] = t[(s, c, m_)]
    return out


def z_tables(path, scenes, models, scale, views=None):
    df = pd.read_csv(path)
    df = df[(df.seed == 0) & (df.split == "test") & df.metric.isin(IMG) & df.variant.isin(["ctrl", "pure", "mixed"])].copy()
    df["model"] = df.loss + "__" + df.variant
    out = np.full((len(scenes), len(models), len(IMG)), np.nan)
    for si, sc in enumerate(scenes):
        g = df[df.scene == sc]
        if views is not None:
            g = g[g.view_id.isin(views[sc])]
        p = g.pivot_table(index=["metric", "view_id"], columns="model", values="value")
        d = p.sub(p[CTRL], axis=0).groupby(level=0).mean()
        for mi, m_ in enumerate(IMG):
            out[si, :, mi] = M.SIGN[m_] * d.loc[m_, models].to_numpy() / scale[mi]
    return out


def self_terms(z, models):
    own = np.stack([z[..., ci, IMG.index(own_metric(c))] for ci, c in enumerate(models)], axis=-1)
    ref = np.stack([z[..., ci, others(own_metric(c), True)].mean(axis=-1) for ci, c in enumerate(models)], axis=-1)
    return own, ref, own - ref


def partial_spearman(x, y, c):
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(c).all()):
        return np.nan
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(c)
    X = np.stack([np.ones_like(rc), rc], axis=1)
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / den) if den > 0 else np.nan


def rho(x, y):
    return float(spearmanr(x, y).statistic)


def boot_stat(fn, n_scene, rng):
    idx = rng.integers(0, n_scene, (B, n_scene))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draws = np.array([fn(i) for i in idx], dtype=float)
        return float(fn(np.arange(n_scene))), float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def main():
    os.makedirs(RES, exist_ok=True)
    rng = np.random.default_rng(0)
    cfg = C.load_cfg(P.code(P.E7, "configs", "pilot.yaml"))
    scenes = list(cfg["scenes"])
    import drift as DR
    lm = DR.loss_models()
    models = [CTRL] + lm
    scale = paper_scale()
    pairs = load_pairs()
    R = {}
    dr = pairs[pairs.kind == "drift"]
    base = dr[dr.model == BASE]
    R["base_check"] = dict(max_abs_distance=float(base.dist.abs().max()), rows=int(len(base)),
                           note="base model = the same checkpoint in both: its two render sets must be identical")
    D = cube(dr, scenes, models) / scale
    Draw = cube(dr, scenes, models, "value")
    rows = []
    for ci, c in enumerate(models):
        for mi, m_ in enumerate(IMG):
            rows.append(dict(model=c, metric=m_, drift_sd=float(D[:, ci, mi].mean()), raw_value=float(Draw[:, ci, mi].mean()),
                             **{f"drift_sd__{s}": float(D[si, ci, mi]) for si, s in enumerate(scenes)}))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "drift_by_model_metric.csv"), index=False)
    w = dr[dr.model != BASE].pivot_table(index=["scene", "model", "view", "metric"], columns="direction", values="dist").reset_index()
    sym = w.groupby("metric").apply(lambda g: pd.Series(dict(mean_ab=g.ab.mean(), mean_ba=g.ba.mean(),
                                                          median_rel_diff=float(np.median(np.abs(g.ab - g.ba) / np.maximum((g.ab.abs() + g.ba.abs()) / 2, 1e-12))))),
                                    include_groups=False).reindex(IMG)
    sym.to_csv(os.path.join(RES, "direction_symmetry.csv"))
    n_views = dr.groupby("scene").view.nunique()
    R["views"] = {s: int(n_views[s]) for s in scenes}
    MagP = cube(pairs[pairs.kind == "change_paper"], scenes, lm) / scale
    MagR = cube(pairs[pairs.kind == "change_rerun"], scenes, lm) / scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Mag = np.nanmean(np.stack([MagP, MagR]), axis=0)
    R["magnitude_from"] = dict(paper_models=bool(np.isfinite(MagP).all()), rerun_models=bool(np.isfinite(MagR).all()))
    sel = {s: DR.held_out(s) for s in scenes}
    fP = os.path.join(P.ORIG_RESULTS, "metrics.csv")
    fR = os.path.join(P.RERUN_RESULTS, "metrics.csv")
    Z = {}
    for tag, views in (("all", None), ("sel", sel)):
        zP, zR = z_tables(fP, scenes, models, scale, views), z_tables(fR, scenes, models, scale, views)
        Z[tag] = dict(paper=zP[:, 1:], rerun=zR[:, 1:], mean=(zP[:, 1:] + zR[:, 1:]) / 2)
    old = pd.read_csv(os.path.join(P.ORIG_TABLES, "self_overestimation_by_scene.csv"))
    _, _, Sp = self_terms(Z["all"]["paper"], lm)
    mx = 0.0
    for si, s in enumerate(scenes):
        for ci, c in enumerate(lm):
            q = old[(old.scene == s) & (old.cond == c.replace("__", "/"))]
            mx = max(mx, abs(float(q.iloc[0].score) - Sp[si, ci]))
    R["S_check_max_abs_diff_vs_paper_table"] = mx
    own_i = [IMG.index(own_metric(c)) for c in lm]
    Dl = D[:, 1:]
    Dc = D[:, :1]
    own_d = np.stack([Dl[:, ci, own_i[ci]] for ci in range(len(lm))], axis=1)
    def other_mean(X, with_psnr=False):
        return np.stack([X[:, ci, others(own_metric(c), with_psnr)].mean(axis=1) for ci, c in enumerate(lm)], axis=1)
    oth = {False: other_mean(Dl), True: other_mean(Dl, True)}
    Drel = Dl / Dc
    own_rel = np.stack([Drel[:, ci, own_i[ci]] for ci in range(len(lm))], axis=1)
    oth_rel = {False: other_mean(Drel), True: other_mean(Drel, True)}
    oth_rel_geo = np.stack([np.exp(np.log(Drel[:, ci, others(own_metric(c))]).mean(axis=1)) for ci, c in enumerate(lm)], axis=1)
    mag_oth = {False: other_mean(Mag), True: other_mean(Mag, True)}
    mag_own = np.stack([Mag[:, ci, own_i[ci]] for ci in range(len(lm))], axis=1)
    n = len(scenes)
    def rel_ratio(i, ci):
        r_ = Dl[i, ci].mean(0) / Dc[i, 0].mean(0)
        return float(r_[own_i[ci]] / np.exp(np.log(r_[others(own_metric(lm[ci]))]).mean()))
    rows = []
    for ci, c in enumerate(lm):
        r = dict(model=c, loss=c.split("__")[0], group=c.split("__")[1], klass=CLASS[c.split("__")[0]], own_metric=own_metric(c),
                 drift_own=float(own_d[:, ci].mean()), drift_other=float(oth[False][:, ci].mean()),
                 drift_other_with_psnr=float(oth[True][:, ci].mean()))
        for tag, num, den in (("ratio", own_d, oth[False]), ("ratio_with_psnr", own_d, oth[True]),
                              ("ratio_rel_ctrl_arith", own_rel, oth_rel[False]), ("ratio_change", mag_own, mag_oth[False])):
            p, lo, hi = boot_stat(lambda i, ci=ci, num=num, den=den: num[i, ci].mean() / den[i, ci].mean(), n, rng)
            r.update({tag: p, tag + "_lo": lo, tag + "_hi": hi})
        p, lo, hi = boot_stat(lambda i, ci=ci: rel_ratio(i, ci), n, rng)
        r.update(ratio_rel_ctrl=p, ratio_rel_ctrl_lo=lo, ratio_rel_ctrl_hi=hi)
        r["scenes_ratio_below_1"] = int((own_d[:, ci] / oth[False][:, ci] < 1).sum())
        r["scenes_ratio_rel_below_1"] = int(sum(rel_ratio(np.array([si]), ci) < 1 for si in range(n)))
        r.update({f"ratio__{s}": float(own_d[si, ci] / oth[False][si, ci]) for si, s in enumerate(scenes)})
        rows.append(r)
    blind = pd.DataFrame(rows)
    blind.to_csv(os.path.join(RES, "blind_ratio.csv"), index=False)
    crow = []
    for k in ("pixel", "structural", "perceptual"):
        for g in ("pure", "mixed"):
            ii = [ci for ci, c in enumerate(lm) if CLASS[c.split("__")[0]] == k and c.endswith(g)]
            p, lo, hi = boot_stat(lambda i, ii=ii: float(np.exp(np.mean(np.log(own_d[i][:, ii].mean(0) / oth[False][i][:, ii].mean(0))))), n, rng)
            crow.append(dict(klass=k, group=g, n_models=len(ii), version="ratio", geometric_mean_ratio=p, lo=lo, hi=hi))
            p, lo, hi = boot_stat(lambda i, ii=ii: float(np.exp(np.mean([np.log(rel_ratio(i, c_)) for c_ in ii]))), n, rng)
            crow.append(dict(klass=k, group=g, n_models=len(ii), version="ratio_rel_ctrl", geometric_mean_ratio=p, lo=lo, hi=hi))
    pd.DataFrame(crow).to_csv(os.path.join(RES, "blind_ratio_by_class.csv"), index=False)
    terms = {tag: {k: self_terms(Z[tag][k], lm) for k in ("paper", "rerun", "mean")} for tag in Z}
    rows = []
    for ci, c in enumerate(lm):
        o, r_, s_ = (x[:, ci].mean() for x in terms["all"]["mean"])
        rows.append(dict(model=c, group=c.split("__")[1], klass=CLASS[c.split("__")[0]], drift_own=own_d[:, ci].mean(),
                         drift_other=oth[False][:, ci].mean(), drift_other_with_psnr=oth[True][:, ci].mean(),
                         drift_other_rel_ctrl=float(np.exp(np.log(oth_rel_geo[:, ci]).mean())), magnitude_other=mag_oth[False][:, ci].mean(),
                         drift_over_magnitude=oth[False][:, ci].mean() / mag_oth[False][:, ci].mean(),
                         S=s_, z_own=o, z_ref=r_, S_paper=terms["all"]["paper"][2][:, ci].mean(),
                         S_rerun=terms["all"]["rerun"][2][:, ci].mean(), S_selected_views=terms["sel"]["mean"][2][:, ci].mean()))
    per_loss = pd.DataFrame(rows)
    per_loss.to_csv(os.path.join(RES, "per_loss.csv"), index=False)
    ps = []
    for si, s in enumerate(scenes):
        for ci, c in enumerate(lm):
            ps.append(dict(scene=s, model=c, drift_own=own_d[si, ci], drift_other=oth[False][si, ci], magnitude_other=mag_oth[False][si, ci],
                           S=terms["all"]["mean"][2][si, ci], z_own=terms["all"]["mean"][0][si, ci], z_ref=terms["all"]["mean"][1][si, ci],
                           S_paper=terms["all"]["paper"][2][si, ci], S_rerun=terms["all"]["rerun"][2][si, ci]))
    pd.DataFrame(ps).to_csv(os.path.join(RES, "per_loss_by_scene.csv"), index=False)
    sets = dict(all=list(range(len(lm))), pure=[i for i, c in enumerate(lm) if c.endswith("pure")],
                mixed=[i for i, c in enumerate(lm) if c.endswith("mixed")])
    rows = []
    for dname, dv in (("other metrics, without PSNR", oth[False]), ("other metrics, with PSNR (60 dB cap)", oth[True]),
                      ("other metrics, relative to the control's drift", oth_rel_geo)):
        for vname, tag in (("all held-out views", "all"), ("selected views", "sel")):
            own_z, ref_z, S_ = terms[tag]["mean"]
            for gname, ii in sets.items():
                for yname, yv in (("S", S_), ("z_own", own_z), ("z_ref", ref_z), ("magnitude", mag_oth[False])):
                    p, lo, hi = boot_stat(lambda i, ii=ii, yv=yv, dv=dv: rho(dv[i][:, ii].mean(0), yv[i][:, ii].mean(0)), n, rng)
                    pv = float(spearmanr(dv[:, ii].mean(0), yv[:, ii].mean(0)).pvalue)
                    rows.append(dict(drift=dname, views=vname, losses=gname, n_losses=len(ii), y=yname, kind="spearman",
                                     value=p, lo=lo, hi=hi, p_over_losses=pv))
                    if yname != "magnitude":
                        p, lo, hi = boot_stat(lambda i, ii=ii, yv=yv, dv=dv: partial_spearman(dv[i][:, ii].mean(0), yv[i][:, ii].mean(0),
                                                                                             mag_oth[False][i][:, ii].mean(0)), n, rng)
                        rows.append(dict(drift=dname, views=vname, losses=gname, n_losses=len(ii), y=yname,
                                         kind="partial spearman (magnitude controlled)", value=p, lo=lo, hi=hi, p_over_losses=np.nan))
                p, lo, hi = boot_stat(lambda i, ii=ii, S_=S_: rho(mag_oth[False][i][:, ii].mean(0), S_[i][:, ii].mean(0)), n, rng)
                rows.append(dict(drift=dname, views=vname, losses=gname, n_losses=len(ii), y="S", kind="spearman of the magnitude (not the drift) with S",
                                 value=p, lo=lo, hi=hi, p_over_losses=float(spearmanr(mag_oth[False][:, ii].mean(0), S_[:, ii].mean(0)).pvalue)))
                p, lo, hi = boot_stat(lambda i, ii=ii, S_=S_, dv=dv: partial_spearman(mag_oth[False][i][:, ii].mean(0), S_[i][:, ii].mean(0),
                                                                                    dv[i][:, ii].mean(0)), n, rng)
                rows.append(dict(drift=dname, views=vname, losses=gname, n_losses=len(ii), y="S",
                                 kind="partial spearman of the magnitude with S (drift controlled)", value=p, lo=lo, hi=hi, p_over_losses=np.nan))
    corr = pd.DataFrame(rows)
    corr.to_csv(os.path.join(RES, "correlations.csv"), index=False)
    S_ = terms["all"]["mean"][2]
    ws = [dict(scene=s, **{g: rho(oth[False][si, ii], S_[si, ii]) for g, ii in sets.items()}) for si, s in enumerate(scenes)]
    pd.DataFrame(ws).to_csv(os.path.join(RES, "correlation_within_scene.csv"), index=False)
    fits = {}
    for gname, ii in sets.items():
        x, y = np.log10(oth[False][:, ii].mean(0)), S_[:, ii].mean(0)
        X = np.stack([np.ones_like(x), x], axis=1)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ beta
        loo = []
        for k in range(len(ii)):
            keep = np.arange(len(ii)) != k
            b_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
            loo.append(float(y[k] - X[k] @ b_))
        fits[gname] = dict(n=len(ii), intercept=float(beta[0]), slope_per_decade=float(beta[1]),
                           r2=float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()), resid_sd=float(np.sqrt((res ** 2).sum() / (len(ii) - 2))),
                           loo_rmse=float(np.sqrt(np.mean(np.square(loo)))), sd_of_S=float(y.std(ddof=1)),
                           x_min=float(x.min()), x_max=float(x.max()), x_mean=float(x.mean()), sxx=float(((x - x.mean()) ** 2).sum()))
    R["fit_S_on_log10_drift"] = fits
    sd_ = pairs[pairs.kind == "seed"]
    if len(sd_):
        ti = scenes.index("gsqa/truck")
        Dseed = cube(sd_, ["gsqa/truck"], models)[0] / scale
        rows = []
        for ci, c in enumerate(models):
            ix = others(own_metric(c)) if c != CTRL else [i for i, k in enumerate(IMG) if k != "psnr"]
            rows.append(dict(model=c, same_checkpoint_twice=float(D[ti, ci, ix].mean()), other_stage_a_seed=float(Dseed[ci, ix].mean()),
                             ratio=float(D[ti, ci, ix].mean() / Dseed[ci, ix].mean()),
                             psnr_same_checkpoint_twice=float(Draw[ti, ci, IMG.index("psnr")]),
                             psnr_other_stage_a_seed=float(cube(sd_, ["gsqa/truck"], models, "value")[0][ci, IMG.index("psnr")])))
        pd.DataFrame(rows).to_csv(os.path.join(RES, "truck_seed_vs_rerun.csv"), index=False)
    spatial(lm, scenes, rng, R)
    colour(lm, scenes, scale, sel, fR, rng, R)
    json.dump(R, open(os.path.join(RES, "results.json"), "w"), indent=1)
    C.log("drift analysis written to " + RES)


def spatial(lm, scenes, rng, R):
    lt = pd.read_parquet(os.path.join(PILOT, "tiles_long.parquet"))
    lt = lt[(lt.split == "test") & (lt.seed == 0)]
    d = A.Data(PILOT, lt=lt)
    maps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "maps", "*.parquet")))], ignore_index=True)
    mi = {(s, c): g.set_index(["view", "tile"]).D for (s, c), g in maps.groupby(["scene", "model"])}
    vals = {}
    for sc in d.scenes:
        meta = d.meta(sc, "test", 0)
        vals[sc] = {m: d.tile_values(sc, "test", 0, m) for m in d.models}
        vals[sc]["_meta"] = meta
    allmean = np.stack([vals[sc][m].mean(0) - vals[sc][CTRL].mean(0) for sc in d.scenes for m in lm])
    scale = np.std(allmean, axis=0, ddof=1)
    def strata_of(sc, m, by):
        meta = vals[sc]["_meta"]
        if by == "U (seed variance of the default loss)":
            return meta.quintile.to_numpy()
        src = mi[(sc, m if by == "own drift (the loss model's two trainings)" else CTRL)]
        v = src.reindex(meta.index).to_numpy()
        return U.assign_strata(v, A.Q)[0]
    rows, sizes = [], []
    BY = ["U (seed variance of the default loss)", "own drift (the loss model's two trainings)", "drift of the control model (its two trainings)"]
    for by in BY:
        per = {g: dict(own=[], ref=[], S=[]) for g in ("pure", "mixed")}
        for sc in d.scenes:
            acc = {g: dict(own=[], ref=[], S=[]) for g in ("pure", "mixed")}
            for m in lm:
                q = strata_of(sc, m, by)
                x, c0 = vals[sc][m], vals[sc][CTRL]
                delta = np.stack([x[q == k].mean(0) - c0[q == k].mean(0) for k in range(1, A.Q + 1)] + [x.mean(0) - c0.mean(0)])
                z = delta / scale
                j = d.metrics.index(own_metric(m))
                own = z[:, j]
                ref = A.ref_z(z, d.metrics, own_metric(m), "paper")
                g = m.split("__")[1]
                acc[g]["own"].append(own), acc[g]["ref"].append(ref), acc[g]["S"].append(own - ref)
            for g in acc:
                for k in acc[g]:
                    per[g][k].append(np.mean(acc[g][k], axis=0))
        for g in per:
            Sx = np.stack(per[g]["S"])
            for k in ("own", "ref", "S"):
                arr = np.stack(per[g][k])
                for gi, name in enumerate(A.STRATA + ["overall"]):
                    p, lo, hi, _ = A.boot_scenes(arr[:, gi], rng)
                    rows.append(dict(strata_by=by, group=g, stratum=name, quantity={"own": "own_z", "ref": "ref_z", "S": "S"}[k],
                                     value=float(p), lo=float(lo), hi=float(hi)))
            diff = Sx[:, A.Q - 1] - Sx[:, 0]
            p, lo, hi, _ = A.boot_scenes(diff, rng)
            rows.append(dict(strata_by=by, group=g, stratum="Q5-Q1", quantity="S", value=float(p), lo=float(lo), hi=float(hi),
                             n_scenes_positive=int((diff > 0).sum()), n_scenes=len(diff)))
            for k in ("own", "ref"):
                arr = np.stack(per[g][k])
                diff = arr[:, A.Q - 1] - arr[:, 0]
                p, lo, hi, _ = A.boot_scenes(diff, rng)
                rows.append(dict(strata_by=by, group=g, stratum="Q5-Q1", quantity=k + "_z", value=float(p), lo=float(lo), hi=float(hi),
                                 n_scenes_positive=int((diff > 0).sum()), n_scenes=len(diff)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "spatial_strata.csv"), index=False)
    rr = []
    for sc in d.scenes:
        meta = vals[sc]["_meta"]
        for m in [CTRL] + lm:
            v = mi[(sc, m)].reindex(meta.index).to_numpy()
            rr.append(dict(scene=sc, model=m, spearman_drift_U=rho(v, meta.U.to_numpy()), median_tile_drift=float(np.median(v)),
                           share_of_squared_drift_in_top20=float(np.sort(v ** 2)[::-1][:max(1, len(v) // 5)].sum() / (v ** 2).sum())))
    pd.DataFrame(rr).to_csv(os.path.join(RES, "spatial_drift_vs_U.csv"), index=False)
    R["spatial_note"] = "held-out tiles of the pilot, loss seed 0 of every scene; tile values of the re-run models; s_m of the tile analysis"


def colour(lm, scenes, scale, sel, f_rerun, rng, R):
    fs = sorted(glob.glob(os.path.join(OUT, "colour", "*.parquet")))
    if not fs:
        return
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    have = [s for s in scenes if s in set(df.scene)]
    models = [CTRL] + lm
    v = {}
    for var in ("none", "affine", "gain"):
        t = df[df.variant == var].groupby(["scene", "model", "metric"]).value.mean()
        v[var] = np.array([[[t.get((s, c, m_), np.nan) for m_ in IMG] for c in models] for s in have])
    ref = pd.read_csv(f_rerun)
    ref = ref[(ref.seed == 0) & (ref.split == "test") & ref.metric.isin(IMG)].copy()
    ref["model"] = ref.loss + "__" + ref.variant
    mx = 0.0
    for si, s in enumerate(have):
        g = ref[(ref.scene == s) & ref.view_id.isin(sel[s])].groupby(["model", "metric"]).value.mean()
        for ci, c in enumerate(models):
            for mi_, m_ in enumerate(IMG):
                mx = max(mx, abs(g[(c, m_)] - v["none"][si, ci, mi_]) / max(abs(g[(c, m_)]), 1e-12))
    R["colour_check_max_rel_diff_unmatched_vs_metric_table"] = float(mx)
    sign = np.array([M.SIGN[m_] for m_ in IMG])
    rows = []
    versions = {"no match": ("none", "none"), "both matched (model and control)": (None, None), "only the loss model matched": (None, "none")}
    for var in ("affine", "gain"):
        for vname, (a, b) in versions.items():
            if vname == "no match" and var == "gain":
                continue
            z = sign * (v[a or var][:, 1:] - v[b or var][:, :1]) / scale
            own, refz, S_ = self_terms(z, lm)
            for ci, c in enumerate(lm):
                if not np.isfinite(S_[:, ci]).all():
                    continue
                for q, arr in (("S", S_), ("z_own", own), ("z_ref", refz)):
                    p, lo, hi, _ = A.boot_scenes(arr[:, ci], rng)
                    rows.append(dict(match=var if vname != "no match" else "-", version=vname, model=c, quantity=q, value=float(p),
                                     lo=float(lo), hi=float(hi), n_scenes=len(have)))
            for g in ("pure", "mixed") if var == "affine" else ():
                ii = [i for i, c in enumerate(lm) if c.endswith(g)]
                p, lo, hi, _ = A.boot_scenes(S_[:, ii].mean(1), rng)
                rows.append(dict(match=var if vname != "no match" else "-", version=vname, model=f"group mean ({g})", quantity="S",
                                 value=float(p), lo=float(lo), hi=float(hi), n_scenes=len(have)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "colour_match_S.csv"), index=False)
    ci = models.index("cvvdp__pure")
    rows = []
    for mi_, m_ in enumerate(IMG):
        rows.append(dict(metric=m_, z_no_match=float((sign * (v["none"][:, ci] - v["none"][:, 0]) / scale)[:, mi_].mean()),
                         z_both_matched_affine=float((sign * (v["affine"][:, ci] - v["affine"][:, 0]) / scale)[:, mi_].mean()),
                         z_both_matched_gain=float((sign * (v["gain"][:, ci] - v["gain"][:, 0]) / scale)[:, mi_].mean())))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "colour_match_cvvdp_pure_by_metric.csv"), index=False)
if __name__ == "__main__":
    main()
