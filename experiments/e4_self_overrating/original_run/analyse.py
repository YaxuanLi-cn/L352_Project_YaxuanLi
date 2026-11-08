#!/usr/bin/env python
import argparse
import glob
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
import numpy as np              
import pandas as pd              
import yaml              
from scipy.stats import kendalltau              
from gh import metrics as M              
B = 2000
IMG = M.IMAGE_METRICS
PATH = ["cvvdp", "cvvdp_sust", "cvvdp_image"]
CTRL = "default/ctrl"
BASE = "none/base"
VARIANTS = ("base", "ctrl", "pure", "mixed")
COMMON = ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg", "flip", "cvvdp_image"]
OLD_CONDITIONS = os.path.join(P.PART1_RESULTS, "conditions.csv")


def load_cfg(path):
    cfg = yaml.safe_load(open(path))
    for k in ("runs_root", "results_dir"):
        cfg[k] = os.path.normpath(os.path.join(P.ROOT, cfg[k]))
    return cfg


def nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def collect(cfg):
    frames, runs = [], []
    for d in sorted(glob.glob(os.path.join(cfg["runs_root"], "*", "s*", "*__*"))):
        if os.path.exists(os.path.join(d, "done.json")):
            info = json.load(open(os.path.join(d, "done.json")))
            frames.append(pd.read_csv(os.path.join(d, "metrics.csv")))
            for f in sorted(glob.glob(os.path.join(d, "extra*.csv"))):
                frames.append(pd.read_csv(f))
            runs.append(dict(info, outcome="done"))
    for f in sorted(glob.glob(os.path.join(cfg["runs_root"] + "_diag", "*", "metrics.csv"))):
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df = df[["scene", "loss", "variant", "seed", "split", "view_id", "metric", "value"]]
    df = df[df.scene.isin(cfg["scenes"])].reset_index(drop=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    df.to_csv(os.path.join(cfg["results_dir"], "metrics.csv"), index=False)
    runs = pd.DataFrame(runs)
    runs = runs[runs.scene.isin(cfg["scenes"])].reset_index(drop=True)
    runs.to_csv(os.path.join(cfg["results_dir"], "runs.csv"), index=False)
    return df, runs


def order_conditions(conds):
    def key(c):
        loss, var = c.split("/")
        return VARIANTS.index(var), (["none", "default"] + list(M.LOSSES)).index(loss)
    return sorted(conds, key=key)


def label(c):
    loss, var = c.split("/")
    if var == "base":
        return "base (stage A only)"
    if var == "ctrl":
        return "ctrl"
    return f"{M.LOSS_LABEL[loss]} ({var})"


class Deltas:
    def __init__(self, df):
        df = df[df.variant.isin(VARIANTS)].copy()
        df["cond"] = df.loss + "/" + df.variant
        self.df = df
        q = df[df.split.isin(["test", "train", "path"]) & (df.metric.isin(IMG) | df.metric.isin(PATH))]
        self.conds = order_conditions([c for c in q.cond.unique() if c != CTRL])
        self.data = {}
        for (scene, seed, split), g in q.groupby(["scene", "seed", "split"]):
            if CTRL not in set(g.cond):
                continue
            mets = IMG if split != "path" else PATH
            piv = g.pivot_table(index=["metric", "view_id"], columns="cond", values="value")
            blocks = []
            for m in mets:
                if m not in piv.index.get_level_values(0):
                    blocks.append(None)
                    continue
                sub = piv.loc[m].sort_index()
                ctrl = sub[CTRL].to_numpy()
                blocks.append(np.stack([M.SIGN[m] * (sub[c].to_numpy() - ctrl) if c in sub
                                        else np.full(len(ctrl), np.nan) for c in self.conds]))
            nv = max(b.shape[1] for b in blocks if b is not None)
            arr = np.full((len(self.conds), len(mets), nv), np.nan)
            for mi, b in enumerate(blocks):
                if b is not None:
                    arr[:, mi, :b.shape[1]] = b
            self.data[(scene, seed, split)] = arr
        self.scenes = sorted({k[0] for k in self.data})
    def scenes_for(self, seed, split):
        return [s for s in self.scenes if (s, seed, split) in self.data]
    def point(self, seed, split, scenes=None):
        scenes = scenes or self.scenes_for(seed, split)
        return np.stack([nanmean(self.data[(s, seed, split)], axis=2) for s in scenes])
    def bootstrap(self, seed, splits, rng, scenes=None, n=B, resample_views=("test", "train")):
        scenes = scenes or self.scenes_for(seed, splits[0])
        nm = {sp: (len(IMG) if sp != "path" else len(PATH)) for sp in splits}
        out = {sp: np.zeros((n, len(self.conds), nm[sp])) for sp in splits}
        for b in range(n):
            pick = rng.integers(0, len(scenes), len(scenes))
            for sp in splits:
                acc = np.zeros(out[sp].shape[1:])
                cnt = np.zeros(out[sp].shape[1:])
                for si in pick:
                    a = self.data.get((scenes[si], seed, sp))
                    if a is None:
                        continue
                    if sp in resample_views:
                        a = a[:, :, rng.integers(0, a.shape[2], a.shape[2])]
                    m = nanmean(a, axis=2)
                    ok = np.isfinite(m)
                    acc += np.where(ok, m, 0.0)
                    cnt += ok
                out[sp][b] = acc / np.where(cnt > 0, cnt, np.nan)
        return out


def ci(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(x, 2.5, axis=axis), np.nanpercentile(x, 97.5, axis=axis)


def finding(val, lo, hi, sd):
    return bool(((lo > 0) or (hi < 0)) and (abs(val) > 2 * sd))
FAMILIES = sorted(set(M.FAMILY[k] for k in IMG))


def _ref(z, ci_, exclude, mode):
    if mode == "metrics":
        idx = [IMG.index(k) for k in IMG if M.FAMILY[k] not in exclude]
        return nanmean(z[..., ci_, idx], axis=-1)
    if mode == "families":
        fam = [nanmean(z[..., ci_, [IMG.index(k) for k in IMG if M.FAMILY[k] == f]], axis=-1)
               for f in FAMILIES if f not in exclude]
        return nanmean(np.stack(fam, axis=-1), axis=-1)
    if mode == "judges":
        return nanmean(z[..., ci_, [IMG.index(k) for k in M.JUDGES]], axis=-1)


def self_scores(z, conds, mode="metrics"):
    out = {}
    for ci_, c in enumerate(conds):
        loss, var = c.split("/")
        if var not in ("pure", "mixed"):
            continue
        j = M.LOSSES[loss]
        fam = M.FAMILY[j]
        ref = _ref(z, ci_, {fam}, mode)
        S = z[..., ci_, IMG.index(j)] - ref
        T = {k: z[..., ci_, IMG.index(k)] - ref for k in IMG if M.FAMILY[k] == fam and k != j}
        out[c] = (S, T)
    return out


def overrating_matrix(z, conds):
    rows = {}
    for ci_, c in enumerate(conds):
        loss, var = c.split("/")
        if var not in ("pure", "mixed"):
            continue
        fam = M.FAMILY[M.LOSSES[loss]]
        rows[c] = np.stack([z[..., ci_, mi] - _ref(z, ci_, {fam, M.FAMILY[k]}, "metrics")
                            for mi, k in enumerate(IMG)], axis=-1)
    return rows


def pair_class(a, b):
    if a.method != b.method:
        return "cross_method"
    full = {(a.ladder, a.level), (b.ladder, b.level)}
    same = a.ladder == b.ladder or (("iterations", 30000) in full and "views" in (a.ladder, b.ladder))
    return "same_ladder" if same else "cross_ladder"


def concordance(dx, dy):
    s = np.sign(dx) * np.sign(dy)
    c, d = int((s > 0).sum()), int((s < 0).sum())
    return ((c - d) / (c + d) if c + d >= 5 else np.nan), c + d


def kendall_analysis(D, tab, rng):
    old = pd.read_csv(OLD_CONDITIONS)
    old = old[old.seed == 42].copy()
    old["scene_key"] = old.dataset + "/" + old.scene
    ft = D.df[(D.df.seed == 0) & (D.df.split == "test") & D.df.metric.isin(COMMON) & (D.df.cond != BASE)]
    ft = ft.groupby(["scene", "cond", "metric"]).value.mean().unstack("metric").reset_index()
    scenes = [s for s in D.scenes if s in set(old.scene_key) and s in set(ft.scene)]
    nat = {s: old[old.scene_key == s].reset_index(drop=True) for s in scenes}
    fin = {s: ft[ft.scene == s].reset_index(drop=True) for s in scenes}
    nm = len(COMMON)
    taus = {}
    for name, sets in (("natural", nat), ("finetune", fin)):
        stack = np.full((len(scenes), nm, nm), np.nan)
        for si, s in enumerate(scenes):
            g = sets[s]
            for a in range(nm):
                for b_ in range(nm):
                    stack[si, a, b_] = kendalltau(g[COMMON[a]] * M.SIGN[COMMON[a]],
                                                  g[COMMON[b_]] * M.SIGN[COMMON[b_]]).statistic
        taus[name] = stack
    draws = rng.integers(0, len(scenes), (B, len(scenes)))
    boot = {k: np.stack([nanmean(v[d], axis=0) for d in draws]) for k, v in taus.items()}
    rows = []
    for name in taus:
        mean = nanmean(taus[name], axis=0)
        lo, hi = ci(boot[name])
        n_cond = int(np.mean([len((nat if name == "natural" else fin)[s]) for s in scenes]))
        for a in range(nm):
            for b_ in range(nm):
                rows.append(dict(set=name, m1=COMMON[a], m2=COMMON[b_], tau=mean[a, b_], lo=lo[a, b_], hi=hi[a, b_],
                                 n_scenes=len(scenes), n_conditions=n_cond))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "kendall_tau.csv"), index=False)
    lo, hi = ci(boot["finetune"] - boot["natural"])
    iu = [(a, b_) for a in range(nm) for b_ in range(a + 1, nm)]
    rows = [dict(m1=COMMON[a], m2=COMMON[b_], natural=nanmean(taus["natural"][:, a, b_]),
                 finetune=nanmean(taus["finetune"][:, a, b_]),
                 diff=nanmean(taus["finetune"][:, a, b_]) - nanmean(taus["natural"][:, a, b_]),
                 lo=lo[a, b_], hi=hi[a, b_]) for a, b_ in iu]
    off = np.array([[boot[k][:, a, b_] for a, b_ in iu] for k in ("natural", "finetune")]).mean(axis=1)
    l2, h2 = ci(off[1] - off[0])
    rows.append(dict(m1="mean of the 21 pairs", m2="", natural=float(np.mean([r["natural"] for r in rows])),
                     finetune=float(np.mean([r["finetune"] for r in rows])),
                     diff=float(np.mean([r["diff"] for r in rows])), lo=l2, hi=h2))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "kendall_tau_diff.csv"), index=False)
    by_scene = [dict(scene=s, set=k, mean_tau=float(np.mean([taus[k][si, a, b_] for a, b_ in iu])))
                for si, s in enumerate(scenes) for k in taus]
    pd.DataFrame(by_scene).to_csv(os.path.join(tab, "kendall_tau_by_scene.csv"), index=False)
    classes = ["same_ladder", "cross_ladder", "cross_method", "finetune"]
    conc = {(c, m): np.full((len(scenes), len(iu)), np.nan) for c in classes for m in (False, True)}
    npairs = {(c, m): np.zeros(len(iu), dtype=int) for c in classes for m in (False, True)}
    for si, s in enumerate(scenes):
        g, f = nat[s], fin[s]
        pi, pj = np.array(list(itertools.combinations(range(len(g)), 2))).T
        cls = np.array([pair_class(g.iloc[i], g.iloc[j]) for i, j in zip(pi, pj)])
        fi, fj = np.array(list(itertools.combinations(range(len(f)), 2))).T
        for k, (a, b_) in enumerate(iu):
            ma, mb = COMMON[a], COMMON[b_]
            ra, rb = np.ptp(f[ma].to_numpy()), np.ptp(f[mb].to_numpy())
            dxa = (g[ma].to_numpy()[pi] - g[ma].to_numpy()[pj]) * M.SIGN[ma]
            dxb = (g[mb].to_numpy()[pi] - g[mb].to_numpy()[pj]) * M.SIGN[mb]
            near = (np.abs(dxa) <= ra) & (np.abs(dxb) <= rb)
            for c in classes[:3]:
                for matched in (False, True):
                    sel = (cls == c) & (near if matched else True)
                    conc[(c, matched)][si, k], n = concordance(dxa[sel], dxb[sel])
                    npairs[(c, matched)][k] += n
            dfa = (f[ma].to_numpy()[fi] - f[ma].to_numpy()[fj]) * M.SIGN[ma]
            dfb = (f[mb].to_numpy()[fi] - f[mb].to_numpy()[fj]) * M.SIGN[mb]
            for matched in (False, True):
                conc[("finetune", matched)][si, k], n = concordance(dfa, dfb)
                npairs[("finetune", matched)][k] += n
    rows = []
    for (c, matched), v in conc.items():
        per_scene = nanmean(v, axis=1)
        bt = np.array([nanmean(per_scene[d]) for d in draws])
        lo, hi = ci(bt)
        rows.append(dict(pair_class=c, range_matched=matched, concordance=float(nanmean(per_scene)), lo=lo, hi=hi,
                         n_scenes=int(np.isfinite(per_scene).sum()), n_pairs_per_metric_pair=float(npairs[(c, matched)].mean())))
        for k, (a, b_) in enumerate(iu):
            rows.append(dict(pair_class=c, range_matched=matched, m1=COMMON[a], m2=COMMON[b_],
                             concordance=float(nanmean(v[:, k])), n_scenes=int(np.isfinite(v[:, k]).sum()),
                             n_pairs_per_metric_pair=float(npairs[(c, matched)][k])))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "concordance_by_cause.csv"), index=False)
    rows = []
    ft_scene = nanmean(conc[("finetune", False)], axis=1)
    for c in classes[:3]:
        for matched in (False, True):
            d_scene = ft_scene - nanmean(conc[(c, matched)], axis=1)
            lo, hi = ci(np.array([nanmean(d_scene[d]) for d in draws]))
            rows.append(dict(comparison=f"finetune - {c}", range_matched=matched, diff=float(nanmean(d_scene)), lo=lo,
                             hi=hi, n_scenes_negative=int((d_scene < 0).sum()), n_scenes=int(np.isfinite(d_scene).sum())))
    for matched in (False, True):
        d_scene = nanmean(conc[("cross_method", matched)], axis=1) - nanmean(conc[("same_ladder", matched)], axis=1)
        lo, hi = ci(np.array([nanmean(d_scene[d]) for d in draws]))
        rows.append(dict(comparison="cross_method - same_ladder", range_matched=matched, diff=float(nanmean(d_scene)),
                         lo=lo, hi=hi, n_scenes_negative=int((d_scene < 0).sum()),
                         n_scenes=int(np.isfinite(d_scene).sum())))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "concordance_by_cause_diff.csv"), index=False)
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    R = cfg["results_dir"]
    tab = os.path.join(R, "tables")
    os.makedirs(tab, exist_ok=True)
    df, runs = collect(cfg)
    D = Deltas(df)
    conds = D.conds
    fine = [i for i, c in enumerate(conds) if c != BASE]
    trained = [c for c in conds if c.split("/")[1] in ("pure", "mixed")]
    rng = np.random.default_rng(0)
    scenes = D.scenes_for(0, "test")
    P0 = {sp: D.point(0, sp) for sp in ("test", "train")}
    def col_scale(p_test):
        s = np.nanstd(p_test[:, fine, :].reshape(-1, len(IMG)), axis=0, ddof=1)
        s[~np.isfinite(s) | (s == 0)] = np.nan
        return s
    scale = col_scale(P0["test"])
    pd.DataFrame(dict(metric=IMG, scale=scale)).to_csv(os.path.join(tab, "column_scale.csv"), index=False)
    bs = D.bootstrap(0, ["test", "train"], rng)
    zb = {sp: bs[sp] / scale for sp in ("test", "train")}
    zp = {sp: nanmean(P0[sp], axis=0) / scale for sp in ("test", "train")}
    seed_scene = cfg.get("seed_scene")
    seeds = [s for s in cfg.get("seed_scene_seeds", [0]) if (seed_scene, s, "test") in D.data]
    raw_seed = {sp: np.stack([D.point(s, sp, [seed_scene])[0] for s in seeds]) for sp in ("test", "train")} \
        if len(seeds) >= 2 else None
    zs = {sp: raw_seed[sp] / scale for sp in raw_seed} if raw_seed else None
    def sd_seed(fn):
        if zs is None:
            return np.nan
        vals = [fn({sp: zs[sp][k] for sp in zs}) for k in range(len(seeds))]
        return float(np.nanstd(np.asarray(vals, dtype=float), ddof=1))
    rows = []
    for sp in ("test", "train"):
        lo, hi = ci(zb[sp])
        rlo, rhi = ci(bs[sp])
        raw = nanmean(P0[sp], axis=0)
        for ci_, c in enumerate(conds):
            for mi, m in enumerate(IMG):
                sds = sd_seed(lambda z, ci_=ci_, mi=mi, sp=sp: z[sp][ci_, mi])
                rows.append(dict(split=sp, cond=c, label=label(c), metric=m, delta=raw[ci_, mi],
                                 delta_lo=rlo[ci_, mi], delta_hi=rhi[ci_, mi], z=zp[sp][ci_, mi],
                                 z_lo=lo[ci_, mi], z_hi=hi[ci_, mi], sd_seed_z=sds,
                                 sd_seed_delta=sds * scale[mi],
                                 n_scenes_better=int((P0[sp][:, ci_, mi] > 0).sum()), n_scenes=len(scenes),
                                 finding=finding(zp[sp][ci_, mi], lo[ci_, mi], hi[ci_, mi], sds)))
    cross = pd.DataFrame(rows)
    cross.to_csv(os.path.join(tab, "cross_matrix.csv"), index=False)
    rows = []
    for sp in ("test", "train"):
        for mode in ("metrics", "families", "judges"):
            Sb = self_scores(zb[sp], conds, mode)
            Sp = self_scores(zp[sp], conds, mode)
            for c in Sp:
                j = M.LOSSES[c.split("/")[0]]
                ci_ = conds.index(c)
                per_scene = self_scores(P0[sp] / scale, conds, mode)[c][0]
                sds = sd_seed(lambda z, c=c, sp=sp, mode=mode: self_scores(z[sp], conds, mode)[c][0])
                lo, hi = ci(Sb[c][0])
                own_lo, own_hi = ci(zb[sp][:, ci_, IMG.index(j)])
                ref_b = zb[sp][:, ci_, IMG.index(j)] - Sb[c][0]
                ref_lo, ref_hi = ci(ref_b)
                rows.append(dict(split=sp, reference=mode, cond=c, label=label(c), metric=j, kind="self",
                                 score=float(Sp[c][0]), lo=lo, hi=hi, sd_seed=sds,
                                 own_z=float(zp[sp][ci_, IMG.index(j)]), own_lo=own_lo, own_hi=own_hi,
                                 ref_z=float(zp[sp][ci_, IMG.index(j)] - Sp[c][0]), ref_lo=ref_lo, ref_hi=ref_hi,
                                 n_scenes_positive=int((per_scene > 0).sum()), n_scenes=len(scenes),
                                 finding=finding(float(Sp[c][0]), lo, hi, sds)))
                for k in Sp[c][1]:
                    sds = sd_seed(lambda z, c=c, k=k, sp=sp, mode=mode: self_scores(z[sp], conds, mode)[c][1][k])
                    lo, hi = ci(Sb[c][1][k])
                    per_scene_k = self_scores(P0[sp] / scale, conds, mode)[c][1][k]
                    own_lo, own_hi = ci(zb[sp][:, ci_, IMG.index(k)])
                    rows.append(dict(split=sp, reference=mode, cond=c, label=label(c), metric=k, kind="same_family",
                                     score=float(Sp[c][1][k]), lo=lo, hi=hi, sd_seed=sds,
                                     own_z=float(zp[sp][ci_, IMG.index(k)]), own_lo=own_lo, own_hi=own_hi,
                                     ref_z=float(zp[sp][ci_, IMG.index(k)] - Sp[c][1][k]),
                                     n_scenes_positive=int((per_scene_k > 0).sum()), n_scenes=len(scenes),
                                     finding=finding(float(Sp[c][1][k]), lo, hi, sds)))
    so = pd.DataFrame(rows)
    so.to_csv(os.path.join(tab, "self_overestimation.csv"), index=False)
    rows = []
    for sp in ("test", "train"):
        Sb = self_scores(zb[sp], conds)
        Sp = self_scores(zp[sp], conds)
        for var in ("pure", "mixed"):
            cs = [c for c in trained if c.endswith("/" + var)]
            bt = np.mean([Sb[c][0] for c in cs], axis=0)
            lo, hi = ci(bt)
            sds = sd_seed(lambda z, cs=cs, sp=sp: np.mean([self_scores(z[sp], conds)[c][0] for c in cs]))
            vals = [float(Sp[c][0]) for c in cs]
            rows.append(dict(split=sp, variant=var, n_models=len(cs), mean=float(np.mean(vals)),
                             median=float(np.median(vals)), min=float(np.min(vals)), max=float(np.max(vals)),
                             lo=lo, hi=hi, sd_seed=sds))
        both = [c.split("/")[0] for c in trained if c.endswith("/mixed") and c.split("/")[0] + "/pure" in trained]
        if both:
            bt = np.mean([Sb[f"{l}/mixed"][0] for l in both], axis=0) / np.mean([Sb[f"{l}/pure"][0] for l in both], axis=0)
            lo, hi = ci(bt)
            rows.append(dict(split=sp, variant="ratio mixed/pure (same losses)", n_models=len(both),
                             mean=float(np.mean([float(Sp[f"{l}/mixed"][0]) for l in both]) /
                                        np.mean([float(Sp[f"{l}/pure"][0]) for l in both])), lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "self_overestimation_summary.csv"), index=False)
    rows = []
    zz_all = P0["test"] / scale
    S_scene = self_scores(zz_all, conds)
    for si, s in enumerate(scenes):
        for c in trained:
            rows.append(dict(scene=s, cond=c, label=label(c), score=float(S_scene[c][0][si])))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "self_overestimation_by_scene.csv"), index=False)
    rows = []
    for si, s in enumerate(scenes if len(scenes) >= 3 else []):
        keep = [k for k in range(len(scenes)) if k != si]
        p = P0["test"][keep]
        zz = nanmean(p, axis=0) / col_scale(p)
        for c, (S, T) in self_scores(zz, conds).items():
            rows.append(dict(left_out=s, cond=c, label=label(c), kind="self", metric=M.LOSSES[c.split("/")[0]],
                             score=float(S)))
            for k, v in T.items():
                rows.append(dict(left_out=s, cond=c, label=label(c), kind="same_family", metric=k, score=float(v)))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "self_overestimation_loso.csv"), index=False)
    rows = []
    if zs is not None:
        for k, sd_ in enumerate(seeds):
            for c, (S, T) in self_scores(zs["test"][k], conds).items():
                rows.append(dict(scene=seed_scene, seed=sd_, cond=c, label=label(c), kind="self",
                                 metric=M.LOSSES[c.split("/")[0]], score=float(S)))
                for kk, v in T.items():
                    rows.append(dict(scene=seed_scene, seed=sd_, cond=c, label=label(c), kind="same_family",
                                     metric=kk, score=float(v)))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "self_overestimation_by_seed.csv"), index=False)
    rows = []
    for sp in ("test", "train"):
        Ob = overrating_matrix(zb[sp], conds)
        Op = overrating_matrix(zp[sp], conds)
        for c in Op:
            lo, hi = ci(Ob[c])
            for mi, k in enumerate(IMG):
                sds = sd_seed(lambda z, c=c, mi=mi, sp=sp: overrating_matrix(z[sp], conds)[c][mi])
                rows.append(dict(split=sp, cond=c, label=label(c), metric=k, score=float(Op[c][mi]), lo=lo[mi],
                                 hi=hi[mi], sd_seed=sds, finding=finding(float(Op[c][mi]), lo[mi], hi[mi], sds)))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "overrating_matrix.csv"), index=False)
    gb = zb["train"] - zb["test"]
    gp = zp["train"] - zp["test"]
    lo, hi = ci(gb)
    rows = []
    for ci_, c in enumerate(conds):
        for mi, m in enumerate(IMG):
            sds = sd_seed(lambda z, ci_=ci_, mi=mi: z["train"][ci_, mi] - z["test"][ci_, mi])
            rows.append(dict(cond=c, label=label(c), metric=m, gap=gp[ci_, mi], lo=lo[ci_, mi], hi=hi[ci_, mi],
                             sd_seed=sds, z_test=zp["test"][ci_, mi], z_train=zp["train"][ci_, mi],
                             own=(c in trained and M.LOSSES[c.split("/")[0]] == m),
                             finding=finding(gp[ci_, mi], lo[ci_, mi], hi[ci_, mi], sds)))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "generalisation_gap.csv"), index=False)
    q = D.df[(D.df.seed == 0) & D.df.split.isin(["test", "train"]) & D.df.metric.isin(IMG)]
    raw = q.groupby(["scene", "cond", "split", "metric"]).value.mean().unstack("split")
    raw["train_minus_test"] = raw["train"] - raw["test"]
    raw = raw.reset_index()
    raw.to_csv(os.path.join(tab, "train_test_raw.csv"), index=False)
    raw.groupby(["cond", "metric"])[["test", "train", "train_minus_test"]].mean().reset_index() \
        .to_csv(os.path.join(tab, "train_test_raw_mean.csv"), index=False)
    vid_scenes = D.scenes_for(0, "path")
    rows = []
    if vid_scenes:
        Pp = D.point(0, "path", vid_scenes)
        Pt = D.point(0, "test", vid_scenes) / scale
        bv = D.bootstrap(0, ["path", "test"], np.random.default_rng(1), scenes=vid_scenes, resample_views=("test",))
        zt = bv["test"] / scale
        seed_path = np.stack([nanmean(D.data[(seed_scene, s, "path")], axis=2) for s in seeds]) \
            if (seed_scene in vid_scenes and len(seeds) >= 2) else None
        def sdp(fn):
            return float(np.std([fn(seed_path[k]) for k in range(len(seeds))], ddof=1)) if seed_path is not None else np.nan
        for ci_, c in enumerate(conds):
            own = IMG.index(M.LOSSES[c.split("/")[0]]) if c in trained else None
            quant = {
                "d_cvvdp_video": (Pp[:, ci_, 0], bv["path"][:, ci_, 0], sdp(lambda a: a[ci_, 0])),
                "d_cvvdp_sust": (Pp[:, ci_, 1], bv["path"][:, ci_, 1], sdp(lambda a: a[ci_, 1])),
                "d_cvvdp_frames": (Pp[:, ci_, 2], bv["path"][:, ci_, 2], sdp(lambda a: a[ci_, 2])),
                "temporal": (Pp[:, ci_, 0] - Pp[:, ci_, 2], bv["path"][:, ci_, 0] - bv["path"][:, ci_, 2],
                             sdp(lambda a: a[ci_, 0] - a[ci_, 2])),
                "transient": (Pp[:, ci_, 0] - Pp[:, ci_, 1], bv["path"][:, ci_, 0] - bv["path"][:, ci_, 1],
                              sdp(lambda a: a[ci_, 0] - a[ci_, 1])),
            }
            vals = dict(cond=c, label=label(c), n_scenes=len(vid_scenes))
            for nm_, (pt, bt, sd) in quant.items():
                lo, hi = ci(bt)
                vals.update({nm_: float(np.mean(pt)), nm_ + "_lo": lo, nm_ + "_hi": hi, nm_ + "_sd_seed": sd,
                             nm_ + "_n_scenes_negative": int((pt < 0).sum()),
                             nm_ + "_finding": finding(float(np.mean(pt)), lo, hi, sd)})
            fz = nanmean(Pt[:, ci_, :], axis=1)
            lo, hi = ci(nanmean(zt[:, ci_, :], axis=1))
            vals.update(frame_metrics_z=float(np.mean(fz)), frame_metrics_z_lo=lo, frame_metrics_z_hi=hi,
                        n_frame_metrics_better=int(np.sum(nanmean(Pt[:, ci_, :], axis=0) > 0)))
            if own is not None:
                lo, hi = ci(zt[:, ci_, own])
                vals.update(own_metric=IMG[own], own_z=float(np.mean(Pt[:, ci_, own])), own_z_lo=lo, own_z_hi=hi)
            vals["frames_better_video_worse"] = bool(vals["frame_metrics_z_lo"] > 0 and vals["d_cvvdp_video_hi"] < 0
                                                     and vals["d_cvvdp_video_finding"])
            vals["own_better_video_worse"] = bool(own is not None and vals["own_z_lo"] > 0
                                                  and vals["d_cvvdp_video_hi"] < 0 and vals["d_cvvdp_video_finding"])
            rows.append(vals)
    pd.DataFrame(rows).to_csv(os.path.join(tab, "video_vs_frames.csv"), index=False)
    rows = []
    for si, s in enumerate(vid_scenes):
        for ci_, c in enumerate(conds):
            rows.append(dict(scene=s, cond=c, label=label(c), d_cvvdp_video=Pp[si, ci_, 0], d_cvvdp_sust=Pp[si, ci_, 1],
                             d_cvvdp_frames=Pp[si, ci_, 2]))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "video_vs_frames_by_scene.csv"), index=False)
    tau_scenes = kendall_analysis(D, tab, rng)
    rows = []
    if raw_seed is not None:
        a = D.df[(D.df.scene == seed_scene) & (D.df.split == "test") & D.df.metric.isin(IMG) & D.df.seed.isin(seeds)]
        absm = a.groupby(["cond", "seed", "metric"]).value.mean().unstack("seed")
        sd_abs = absm.std(axis=1, ddof=1).unstack("metric")
        sd_delta = np.std(raw_seed["test"], axis=0, ddof=1)
        for mi, m in enumerate(IMG):
            rows.append(dict(metric=m, scale=scale[mi],
                             sd_abs_ctrl=float(sd_abs.loc[CTRL, m]), sd_abs_median=float(sd_abs[m].median()),
                             sd_delta_median=float(np.median(sd_delta[fine, mi])),
                             sd_delta_max=float(np.max(sd_delta[fine, mi])),
                             sd_delta_median_z=float(np.median(sd_delta[fine, mi]) / scale[mi]),
                             sd_delta_max_z=float(np.max(sd_delta[fine, mi]) / scale[mi])))
    pd.DataFrame(rows).to_csv(os.path.join(tab, "seed_noise.csv"), index=False)
    keys = ["scene", "seed", "loss", "variant"]
    st = df[(df.split == "model") & df.variant.isin(VARIANTS)].pivot_table(index=keys, columns="metric",
                                                                         values="value").reset_index()
    st = st.merge(runs.drop(columns=[c for c in runs.columns if c in st.columns and c not in keys]), on=keys, how="outer")
    st["gaussians_equal_base"] = st.num_gaussians == st.base_num_gaussians
    st.to_csv(os.path.join(tab, "runs_status.csv"), index=False)
    hours = dict(stage_b_train_h=float(st.train_time.sum() / 3600), eval_h=float(st.eval_time.sum() / 3600),
                 n_runs=int(len(st)), n_done=int((st.outcome == "done").sum()),
                 n_failed=int((st.outcome == "failed").sum()),
                 n_not_ok=int(((st.outcome == "done") & (st.status != "ok")).sum()),
                 n_gaussians_changed=int(((st.outcome == "done") & ~st.gaussians_equal_base).sum()),
                 n_incomplete=int(((st.outcome == "done") & (st.variant != "base") &
                                   (st.steps_done != st.stage_b_steps)).sum()))
    b = df[df.metric.str.startswith("diag_bias") & (df.split == "test") & df.variant.isin(VARIANTS)].copy()
    b["cond"] = b.loss + "/" + b.variant
    pv = b.pivot_table(index=["scene", "seed", "cond", "view_id"], columns="metric", values="value").reset_index()
    pv["brightness"] = pv[["diag_bias_r", "diag_bias_g", "diag_bias_b"]].mean(axis=1)
    pv["cast"] = pv[["diag_bias_r", "diag_bias_g", "diag_bias_b"]].max(axis=1) - \
        pv[["diag_bias_r", "diag_bias_g", "diag_bias_b"]].min(axis=1)
    cv = pv[pv.cond == CTRL].set_index(["scene", "seed", "view_id"])[["diag_bias_r", "diag_bias_g", "diag_bias_b"]]
    pv = pv.join(cv.add_suffix("_ctrl"), on=["scene", "seed", "view_id"])
    pv["shift_sq"] = sum((pv[f"diag_bias_{ch}"] - pv[f"diag_bias_{ch}_ctrl"]) ** 2 for ch in "rgb") / 3
    per = pv.groupby(["scene", "seed", "cond"])[["diag_bias_r", "diag_bias_g", "diag_bias_b", "brightness", "cast",
                                                 "shift_sq"]].mean().reset_index()
    per["drift_rms"] = np.sqrt(per.shift_sq)
    ctrl_b = per[per.cond == CTRL].set_index(["scene", "seed"])[["brightness", "cast"]].add_suffix("_ctrl")
    per = per.join(ctrl_b, on=["scene", "seed"])
    per["d_brightness"] = per.brightness - per.brightness_ctrl
    per["d_cast"] = per.cast - per.cast_ctrl
    per.to_csv(os.path.join(tab, "colour_bias_by_scene.csv"), index=False)
    s0 = per[per.seed == 0].groupby("cond")[["brightness", "cast", "d_brightness", "d_cast", "drift_rms"]].agg(["mean", "min", "max"])
    s0.columns = ["_".join(c) for c in s0.columns]
    sd = per[(per.scene == seed_scene)].groupby("cond")[["d_brightness", "d_cast", "drift_rms"]].std(ddof=1).add_suffix("_sd_seed")
    s0 = s0.join(sd).reindex([CTRL] + conds).reset_index()
    s0["label"] = s0.cond.map(label)
    s0.to_csv(os.path.join(tab, "colour_bias.csv"), index=False)
    ws = []
    for f in sorted(glob.glob(os.path.join(cfg["runs_root"], "*", "s*", "mixed_weights.json"))):
        parts = f.split(os.sep)
        for loss, v in json.load(open(f)).items():
            ws.append(dict(scene=parts[-3], seed=int(parts[-2][1:]), loss=loss, weight=v["weight"],
                           default_mean=v["default_mean"], metric_loss_mean=v["metric_loss_mean"]))
    ws = pd.DataFrame(ws).rename(columns={"scene": "scene_dir", "weight": "weight_file"})
    used = st[(st.variant == "mixed")][["scene", "seed", "loss", "mixed_weight"]].rename(columns={"mixed_weight": "weight"})
    used["scene_dir"] = used.scene.str.replace("/", "_")
    used = used.merge(ws, on=["scene_dir", "seed", "loss"], how="left").drop(columns="scene_dir")
    used.to_csv(os.path.join(tab, "mixed_weights.csv"), index=False)
    hours["mixed_weight_max_rel_dev"] = float(((used.weight - used.weight_file).abs() / used.weight).max())
    hours["mixed_weight_missing_in_file"] = int(used.weight_file.isna().sum())
    json.dump(hours, open(os.path.join(tab, "run_summary.json"), "w"), indent=1)
    base = df[(df.variant == "base") & (df.split.isin(["test", "model"]))].groupby(["scene", "seed", "metric"]).value.mean()
    base.unstack("metric").reset_index().to_csv(os.path.join(tab, "baseline_values.csv"), index=False)
    bp = df[(df.variant == "base") & (df.split == "path")].groupby(["scene", "seed", "metric"]).value.mean()
    bp.unstack("metric").reset_index().to_csv(os.path.join(tab, "baseline_path_values.csv"), index=False)
    dg = df[df.variant.str.startswith("diag:")]
    if len(dg):
        dg.groupby(["scene", "variant", "seed", "metric"]).value.mean().unstack("metric").reset_index() \
            .to_csv(os.path.join(tab, "baseline_diagnostics.csv"), index=False)
    absval = df[df.split.isin(["test", "train", "path"]) & ~df.metric.str.startswith("diag_") & df.variant.isin(VARIANTS)]
    absval.groupby(["scene", "seed", "loss", "variant", "split", "metric"]).value.mean().reset_index() \
        .to_csv(os.path.join(tab, "absolute_means.csv"), index=False)
    json.dump(dict(conditions=conds, scenes=D.scenes, video_scenes=vid_scenes, tau_scenes=tau_scenes,
                   seed_scene=seed_scene, seeds=seeds, bootstrap=B,
                   n_test_views={s: int(D.data[(s, 0, "test")].shape[2]) for s in scenes},
                   n_train_views={s: int(D.data[(s, 0, "train")].shape[2]) for s in scenes}),
              open(os.path.join(tab, "meta.json"), "w"), indent=1)
    print("analysis written to", tab)
if __name__ == "__main__":
    main()
