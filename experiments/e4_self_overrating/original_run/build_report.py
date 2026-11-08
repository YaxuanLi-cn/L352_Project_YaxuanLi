#!/usr/bin/env python
import argparse
import json
import os
import re
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
ROOT = P.ROOT
import numpy as np              
import pandas as pd              
import yaml              
from gh import metrics as M              
CTX = {}
CTRL = "default/ctrl"
SCENE_SHORT = lambda s: s.split("/")[1]              


def setup(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    R = os.path.normpath(os.path.join(P.ROOT, cfg["results_dir"]))
    CTX.update(cfg=cfg, R=R, T=os.path.join(R, "tables"), cache={}, rel=os.path.relpath(R, P.ROOT))


def T(name):
    c = CTX["cache"]
    if name not in c:
        base = os.path.join(CTX["T"], name)
        if os.path.exists(base + ".json"):
            c[name] = json.load(open(base + ".json"))
        else:
            try:
                c[name] = pd.read_csv(base + ".csv")
            except pd.errors.EmptyDataError:
                c[name] = pd.DataFrame()
    return c[name]


def CFG(key):
    return CTX["cfg"][key]


def f(x, d=2):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{d}f}"


def fs(x, d=2):
    return "n/a" if not np.isfinite(x) else f"{x:+.{d}f}"


def iv(lo, hi, d=2):
    return f"[{fs(lo, d)}, {fs(hi, d)}]"


def big(x):
    return f"{int(round(x)):,}"


def L(m):
    return M.LABEL.get(m, m)


def lab(c):
    loss, var = c.split("/")
    if var == "base":
        return "base (stage A only)"
    if var == "ctrl":
        return "ctrl"
    return f"{M.LOSS_LABEL[loss]} ({var})"


def md(df, cols=None, headers=None):
    cols = cols or list(df.columns)
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join([":--"] + ["--:"] * (len(cols) - 1)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def one(q, col):
    return q[col].iloc[0]


def X(cond, metric, col="z", split="test"):
    t = T("cross_matrix")
    return one(t[(t.cond == cond) & (t.metric == metric) & (t.split == split)], col)


def S(cond, col="score", split="test", metric=None, reference="metrics"):
    t = T("self_overestimation")
    q = t[(t.cond == cond) & (t.split == split) & (t.reference == reference)]
    q = q[q.kind == "self"] if metric is None else q[(q.kind == "same_family") & (q.metric == metric)]
    return one(q, col)


def SS(variant, col="mean", split="test"):
    t = T("self_overestimation_summary")
    return one(t[(t.variant.str.startswith(variant)) & (t.split == split)], col)


def G(cond, metric=None, col="gap"):
    t = T("generalisation_gap")
    q = t[t.cond == cond]
    q = q[q.own] if metric is None else q[q.metric == metric]
    return one(q, col)


def V(cond, col):
    t = T("video_vs_frames")
    return one(t[t.cond == cond], col)


def TAU(m1, m2, set_, col="tau"):
    t = T("kendall_tau")
    return one(t[(t.m1 == m1) & (t.m2 == m2) & (t.set == set_)], col)


def TD(m1, m2=None, col="diff"):
    t = T("kendall_tau_diff")
    q = t[t.m1 == m1] if m2 is None else t[(t.m1 == m1) & (t.m2 == m2)]
    return one(q, col)


def CC(cls, matched, col="concordance"):
    t = T("concordance_by_cause")
    return one(t[(t.pair_class == cls) & (t.range_matched == matched) & t.m1.isna()], col)


def CCD(comparison, matched, col="diff"):
    t = T("concordance_by_cause_diff")
    return one(t[(t.comparison == comparison) & (t.range_matched == matched)], col)


def OM(cond, metric, col="score", split="test"):
    t = T("overrating_matrix")
    return one(t[(t.cond == cond) & (t.metric == metric) & (t.split == split)], col)


def BV(scene, metric, seed=0):
    t = T("baseline_values")
    return one(t[(t.scene == scene) & (t.seed == seed)], metric)


def BPV(scene, metric, seed=0):
    t = T("baseline_path_values")
    return one(t[(t.scene == scene) & (t.seed == seed)], metric)


def n_worse(cond, split="test"):
    own = M.LOSSES[cond.split("/")[0]]
    return sum(X(cond, m, "z", split) < 0 for m in M.IMAGE_METRICS if m != own)


def PUB(source, scene, col="psnr"):
    c = CTX["cache"]
    if "pub" not in c:
        c["pub"] = pd.read_csv(os.path.join(HERE, "configs", "published_baselines.csv"))
    t = c["pub"]
    return one(t[(t.source == source) & (t.scene == scene)], col)


def DIAG(name, metric):
    t = T("baseline_diagnostics")
    return one(t[t.variant.str.contains(name, regex=False)], metric)


def RS(key):
    return T("run_summary")[key]


def META(key):
    return T("meta")[key]


def SN(metric, col):
    t = T("seed_noise")
    return one(t[t.metric == metric], col)


def CB(cond, col):
    t = T("colour_bias")
    return one(t[t.cond == cond], col)


def MEANABS(cond, metric, split="test"):
    t = T("absolute_means")
    loss, var = cond.split("/")
    q = t[(t.loss == loss) & (t.variant == var) & (t.metric == metric) & (t.split == split) & (t.seed == 0)]
    return float(q.value.mean())


def ABS(scene, cond, metric, split="test", seed=0):
    t = T("absolute_means")
    loss, var = cond.split("/")
    return one(t[(t.scene == scene) & (t.loss == loss) & (t.variant == var) & (t.metric == metric) &
                 (t.split == split) & (t.seed == seed)], "value")


def UT(name, col="max_abs_diff"):
    c = CTX["cache"]
    if "ut" not in c:
        c["ut"] = json.load(open(os.path.join(CTX["R"], "unit_tests.json")))
    return c["ut"]["results"][name][col]


def BENCH(loss, col="ms"):
    c = CTX["cache"]
    if "bench" not in c:
        c["bench"] = json.load(open(os.path.join(CTX["R"], "bench_1039x1558.json")))
    return c["bench"][loss][col]


def HOURS(what):
    t = T("runs_status")
    if what == "stage_a":
        return float(t.stage_a_train_time_s.sum() / 3600)
    if what == "stage_b":
        return float(t.train_time_s.sum() / 3600)
    if what == "eval":
        return float(t.eval_time_s.sum() / 3600)


def OLD(scene, col="psnr"):
    c = CTX["cache"]
    if "old" not in c:
        c["old"] = pd.read_csv(os.path.join(P.PART1_RESULTS, "conditions.csv"))
    t = c["old"]
    ds, sc = scene.split("/")
    return one(t[(t.dataset == ds) & (t.scene == sc) & (t.method == "3dgs") & (t.condition == "iters-30000")], col)


def RERUN(scene, col, thresh=3.0):
    c = CTX["cache"]
    key = ("rerun", scene, thresh)
    if key not in c:
        old = np.array(json.load(open(os.path.join(P.PART1_RENDERS, scene, "3dgs", "iters-30000",
                                                   "metrics.json")))["frames"]["psnr"])
        df = c.setdefault("long", pd.read_csv(os.path.join(CTX["R"], "metrics.csv")))
        q = df[(df.scene == scene) & (df.variant == "base") & (df.seed == 0) & (df.split == "test") & (df.metric == "psnr")]
        new = q.sort_values("view_id").value.to_numpy()
        d = new - old
        bigm = np.abs(d) > thresh
        c[key] = dict(old=float(old.mean()), new=float(new.mean()), n_big=int(bigm.sum()), n=int(len(d)),
                      n_big_down=int((d < -thresh).sum()), n_big_up=int((d > thresh).sum()),
                      rest=float(d[~bigm].mean()), min=float(d.min()), max=float(d.max()),
                      old_median=float(np.median(old)), new_median=float(np.median(new)), sd=float(d.std()))
    return c[key][col]


def WALL(what="hours"):
    import datetime
    ts = []
    for w in (1, 2):
        for line in open(P.rec(P.E4, "original_run", "logs", f"full_w{w}.log"), errors="ignore"):
            m = re.match(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) ", line)
            if m:
                mo, da, h, mi, se = map(int, m.groups())
                ts.append(datetime.datetime(2026, mo, da, h, mi, se))
    ts = sorted(ts)
    gaps = np.array([(b - a).total_seconds() for a, b in zip(ts[:-1], ts[1:])])
    if what == "hours":
        return float(gaps[gaps < 3600].sum() / 3600)
    if what == "interruptions":
        return int((gaps >= 3600).sum())
    if what == "lost_hours":
        return float(gaps[gaps >= 3600].sum() / 3600)


def n_self(split="test", reference="metrics", what="finding"):
    t = T("self_overestimation")
    q = t[(t.split == split) & (t.reference == reference) & (t.kind == "self")]
    if what == "finding":
        return int((q.finding & (q.score > 0)).sum())
    if what == "n":
        return len(q)
    if what == "min_scenes":
        return int(q.n_scenes_positive.min())


def n_own_gain(split="test", variant="pure"):
    t = T("cross_matrix")
    n = 0
    for c in META("conditions"):
        loss, var = c.split("/")
        if var == variant:
            q = t[(t.cond == c) & (t.metric == M.LOSSES[loss]) & (t.split == split)]
            n += int(bool(q.finding.iloc[0]) and q.z.iloc[0] > 0)
    return n


def n_own_positive(split="test"):
    return sum(S(c, "own_z", split) > 0 for c in trained())


def TS(scene, set_):
    t = T("kendall_tau_by_scene")
    return one(t[(t.scene == scene) & (t.set == set_)], "mean_tau")


def n_tau(what):
    t = T("kendall_tau_diff")
    q = t[t.m2.notna() & (t.m2 != "")]
    q = q[~q.m1.str.startswith("mean")]
    if what == "pairs":
        return len(q)
    if what == "lower":
        return int((q["diff"] < 0).sum())
    if what == "ci_below":
        return int((q.hi < 0).sum())


def n_cross(split, sign=0):
    t = T("cross_matrix")
    q = t[(t.split == split) & (t.cond != "none/base")]
    if sign > 0:
        return int((q.finding & (q.z > 0)).sum())
    if sign < 0:
        return int((q.finding & (q.z < 0)).sum())
    return len(q)


def n_loso(kind="self"):
    t = T("self_overestimation_loso")
    return float(t[t.kind == kind].score.min())


def loso_range(cond, metric=None):
    t = T("self_overestimation_loso")
    if not len(t):
        return "n/a"
    q = t[(t.cond == cond) & (t.kind == ("self" if metric is None else "same_family"))]
    if metric is not None:
        q = q[q.metric == metric]
    return f"{f(q.score.min())} to {f(q.score.max())}"


def seed_range(cond, metric=None):
    t = T("self_overestimation_by_seed")
    if not len(t):
        return "n/a"
    q = t[(t.cond == cond) & (t.kind == ("self" if metric is None else "same_family"))]
    if metric is not None:
        q = q[q.metric == metric]
    return f"{f(q.score.min())} to {f(q.score.max())}"


def trained():
    return [c for c in META("conditions") if c.split("/")[1] in ("pure", "mixed")]


def tab_baseline():
    rows = []
    for s in [x for x in CFG("scenes") if x.startswith("mip360/")]:
        d = dict(scene=SCENE_SHORT(s), N=f"{BV(s, 'num_gaussians') / 1e6:.2f} M", psnr=f(BV(s, "psnr")),
                 ssim=f(BV(s, "ssim"), 3), lp_vgg=f(BV(s, "lpips_vgg"), 3), lp_01=f(BV(s, "lpips_vgg_01input"), 3),
                 lp_alex=f(BV(s, "lpips_alex"), 3))
        d["kerbl"] = f"{f(PUB('kerbl2023', s))} / {f(PUB('kerbl2023', s, 'ssim'), 3)} / {f(PUB('kerbl2023', s, 'lpips'), 3)}"
        g = "gsplat_docs_gsplat30k"
        d["gsplat"] = f"{f(PUB(g, s))} / {f(PUB(g, s, 'ssim'), 3)} / {f(PUB(g, s, 'lpips'), 3)} ({f(PUB(g, s, 'num_gaussians_m'))} M)"
        i = "gsplat_docs_inria30k"
        d["inria"] = f"{f(PUB(i, s))} / {f(PUB(i, s, 'ssim'), 3)} / {f(PUB(i, s, 'lpips'), 3)}"
        d["d_kerbl"] = fs(BV(s, "psnr") - PUB("kerbl2023", s))
        d["d_gsplat"] = fs(BV(s, "psnr") - PUB(g, s))
        rows.append(d)
    return md(pd.DataFrame(rows),
              ["scene", "N", "psnr", "ssim", "lp_vgg", "lp_01", "lp_alex", "kerbl", "inria", "gsplat", "d_kerbl", "d_gsplat"],
              ["scene", "Gaussians", "PSNR", "SSIM", "LPIPS-VGG", "LPIPS-VGG, [0,1] input", "LPIPS-Alex",
               "Kerbl et al.: PSNR / SSIM / LPIPS (VGG, [0,1])", "Inria code re-run: PSNR / SSIM / LPIPS (Alex)",
               "gsplat: PSNR / SSIM / LPIPS (Alex) (Gaussians)", "PSNR - Kerbl", "PSNR - gsplat"])


def tab_runs_by_scene():
    t = T("runs_status")
    rows = []
    for (scene, seed), g in t.groupby(["scene", "seed"], sort=False):
        b = g[g.variant == "base"].iloc[0]
        ft = g[g.variant != "base"]
        rows.append(dict(scene=scene, seed=seed, runs=len(g), done=int((g.outcome == "done").sum()),
                         ok=int((g.status == "ok").sum()), steps=f"{int(ft.steps_done.min())}-{int(ft.steps_done.max())}",
                         N=big(b.num_gaussians), same=int(g.gaussians_equal_base.sum()),
                         a=f(b.stage_a_train_time_s / 60, 1), bt=f(ft.train_time_s.sum() / 3600, 2)))
    return md(pd.DataFrame(rows), ["scene", "seed", "runs", "done", "ok", "steps", "N", "same", "a", "bt"],
              ["scene", "seed", "runs", "finished", "status ok", "fine-tuning steps (min-max)", "Gaussians (base)",
               "runs with the same count", "stage A (min)", "stage B, all runs (h)"])


def tab_cross_raw(metrics=("psnr", "ssim", "lpips_vgg", "lpips_alex", "dists", "flip", "cvvdp_image", "dreamsim", "fsim", "vif"),
                  split="test"):
    dec = {"psnr": 2, "cvvdp_image": 2, "ssim": 3, "ms_ssim": 3, "lpips_vgg": 3, "lpips_alex": 3, "dists": 3, "flip": 3,
           "dreamsim": 4, "fsim": 3, "vif": 3, "l1": 4, "mse": 5}
    rows = []
    for c in [CTRL] + META("conditions"):
        d = dict(model=lab(c))
        for m in metrics:
            if c == CTRL:
                d[m] = f(MEANABS(c, m, split), dec[m])
            else:
                d[m] = fs(X(c, m, "delta", split), dec[m]) + ("*" if X(c, m, "finding", split) else "")
        rows.append(d)
    return md(pd.DataFrame(rows), ["model"] + list(metrics), ["model"] + [L(m) for m in metrics])


def tab_self(split="test"):
    rows = []
    for c in trained():
        rows.append(dict(model=lab(c), metric=L(M.LOSSES[c.split("/")[0]]),
                         own=fs(S(c, "own_z", split)) + " " + iv(S(c, "own_lo", split), S(c, "own_hi", split)),
                         ref=fs(S(c, "ref_z", split)) + " " + iv(S(c, "ref_lo", split), S(c, "ref_hi", split)),
                         s=fs(S(c, "score", split)) + " " + iv(S(c, "lo", split), S(c, "hi", split)),
                         sd=f(2 * S(c, "sd_seed", split)),
                         sc=f"{S(c, 'n_scenes_positive', split)}/{S(c, 'n_scenes', split)}",
                         fnd="yes" if S(c, "finding", split) else "no"))
    return md(pd.DataFrame(rows), ["model", "metric", "own", "ref", "s", "sd", "sc", "fnd"],
              ["model (loss)", "its metric j", "z_j [95% CI]", "mean z of other families [95% CI]", "self score S [95% CI]",
               "2 x seed SD", "scenes with S > 0", "counts"])


def tab_self_variants():
    rows = []
    for c in trained():
        d = dict(model=lab(c))
        for sp in ("test", "train"):
            for ref in ("metrics", "families", "judges"):
                d[f"{sp}_{ref}"] = fs(S(c, "score", sp, reference=ref)) + ("*" if S(c, "finding", sp, reference=ref) else "")
        d["loso"] = loso_range(c)
        d["seeds"] = seed_range(c)
        rows.append(d)
    return md(pd.DataFrame(rows), ["model", "test_metrics", "test_families", "test_judges", "train_metrics",
                                   "train_families", "train_judges", "loso", "seeds"],
              ["model (loss)", "test: all other-family metrics", "test: families weighted equally",
               "test: DreamSim, FSIM, VIF only", "train: all", "train: families", "train: judges",
               "test, one scene left out (range)", "truck, seeds 0-2 (range)"])


def tab_transfer(split="test"):
    t = T("self_overestimation")
    q = t[(t.split == split) & (t.reference == "metrics") & (t.kind == "same_family")]
    rows = []
    for _, r in q.iterrows():
        rows.append(dict(model=r.label, metric=L(r.metric), z=fs(r.own_z) + " " + iv(r.own_lo, r.own_hi),
                         t=fs(r.score) + " " + iv(r.lo, r.hi), sd=f(2 * r.sd_seed),
                         sc=f"{r.n_scenes_positive}/{r.n_scenes}",
                         fnd=("overrated" if r.score > 0 else "underrated") if r.finding else "no",
                         loso=loso_range(r.cond, r.metric)))
    return md(pd.DataFrame(rows), ["model", "metric", "z", "t", "sd", "sc", "fnd", "loso"],
              ["model (loss)", "evaluated with", "z of that metric [95% CI]", "transfer score T [95% CI]", "2 x seed SD",
               "scenes with T > 0", "finding", "one scene left out (range)"])


def tab_gap_own():
    t = T("generalisation_gap")
    q = t[t.own]
    rows = []
    for _, r in q.iterrows():
        share = f"{100 * r.z_test / r.z_train:.0f} %" if (r.z_train > 0 and r.z_test > 0) else "none"
        rows.append(dict(model=r.label, metric=L(r.metric), tr=fs(r.z_train), te=fs(r.z_test),
                         gap=fs(r.gap) + " " + iv(r.lo, r.hi), sd=f(2 * r.sd_seed), share=share,
                         fnd="yes" if r.finding else "no"))
    return md(pd.DataFrame(rows), ["model", "metric", "tr", "te", "gap", "sd", "share", "fnd"],
              ["model (loss)", "its metric", "z, training views", "z, held-out views", "gap [95% CI]", "2 x seed SD",
               "held-out / training gain", "counts"])


def tab_video():
    t = T("video_vs_frames")
    rows = []
    for _, r in t.iterrows():
        rows.append(dict(model=r.label,
                         own="" if not isinstance(r.get("own_metric"), str) else f"{fs(r.own_z)} {iv(r.own_z_lo, r.own_z_hi)}",
                         fz=f"{fs(r.frame_metrics_z)} {iv(r.frame_metrics_z_lo, r.frame_metrics_z_hi)}",
                         nb=f"{r.n_frame_metrics_better}/{len(M.IMAGE_METRICS)}",
                         vid=f"{fs(r.d_cvvdp_video, 3)} {iv(r.d_cvvdp_video_lo, r.d_cvvdp_video_hi, 3)}"
                             + ("*" if r.d_cvvdp_video_finding else ""),
                         sd=f(2 * r.d_cvvdp_video_sd_seed, 3), neg=f"{r.d_cvvdp_video_n_scenes_negative}/{r.n_scenes}",
                         fr=fs(r.d_cvvdp_frames, 3), tr=f"{fs(r.transient, 3)} {iv(r.transient_lo, r.transient_hi, 3)}",
                         flag=("own metric better, video worse" if r.own_better_video_worse else "")
                         + ("; all frames better, video worse" if r.frames_better_video_worse else "")))
    return md(pd.DataFrame(rows), ["model", "own", "fz", "nb", "vid", "sd", "neg", "fr", "tr", "flag"],
              ["model (loss)", "z of its own metric, test frames [CI]", "mean z of the 13 per-frame metrics [CI]",
               "per-frame metrics better", "CVVDP video, JOD vs ctrl [CI]", "2 x seed SD", "scenes worse",
               "CVVDP mean of frames, JOD", "transient channel: video - sustained only, JOD [CI]", "flag"])


def tab_tau_diff():
    t = T("kendall_tau_diff")
    rows = []
    for _, r in t.iterrows():
        name = r.m1 if r.m1.startswith("mean") else f"{L(r.m1)} - {L(r.m2)}"
        rows.append(dict(pair=name, nat=f(r.natural), ft=f(r.finetune), d=fs(r["diff"]) + " " + iv(r.lo, r.hi)))
    return md(pd.DataFrame(rows), ["pair", "nat", "ft", "d"],
              ["metric pair", "natural degradations", "loss fine-tuning", "difference [95% CI]"])


def tab_cc():
    names = {"same_ladder": "same degradation ladder", "cross_ladder": "different ladders, same method",
             "cross_method": "3DGS vs NeRF", "finetune": "different training losses (this experiment)"}
    rows = []
    for c in names:
        d = dict(cause=names[c], a=f"{f(CC(c, False))} {iv(CC(c, False, 'lo'), CC(c, False, 'hi'))}",
                 n_a=f(CC(c, False, "n_pairs_per_metric_pair"), 0))
        if c == "finetune":
            d.update(m="(same)", n_m="", dd="")
        else:
            d.update(m=f"{f(CC(c, True))} {iv(CC(c, True, 'lo'), CC(c, True, 'hi'))}",
                     n_m=f(CC(c, True, "n_pairs_per_metric_pair"), 0),
                     dd=f"{fs(CCD('finetune - ' + c, True))} {iv(CCD('finetune - ' + c, True, 'lo'), CCD('finetune - ' + c, True, 'hi'))}")
        rows.append(d)
    return md(pd.DataFrame(rows), ["cause", "a", "n_a", "m", "n_m", "dd"],
              ["what makes the two models differ", "concordance, all pairs [95% CI]", "pairs", "range-matched pairs [95% CI]",
               "pairs", "fine-tuning minus this class, range-matched [95% CI]"])


def tab_colour():
    t = T("colour_bias")
    rows = []
    for _, r in t.iterrows():
        if r.cond == CTRL:
            continue
        rows.append(dict(model=r.label, drift=f"{f(r.drift_rms_mean)} ({f(r.drift_rms_min)}-{f(r.drift_rms_max)})",
                         sd=f(r.drift_rms_sd_seed), b=f"{fs(r.d_brightness_mean)} ({fs(r.d_brightness_min)} to {fs(r.d_brightness_max)})",
                         c=f"{fs(r.d_cast_mean)} ({fs(r.d_cast_min)} to {fs(r.d_cast_max)})",
                         psnr=fs(X(r.cond, "psnr", "delta"))))
    return md(pd.DataFrame(rows), ["model", "drift", "sd", "b", "c", "psnr"],
              ["model", "per-view colour shift vs ctrl, RMS (range over scenes)", "seed SD (truck)",
               "mean brightness change (range)", "change of the spread between R, G, B (range)", "PSNR change (dB)"])


def tab_seed_noise():
    t = T("seed_noise")
    dec = {"psnr": 3, "cvvdp_image": 3}
    rows = []
    for _, r in t.iterrows():
        d = dec.get(r.metric, 5 if r.metric in ("mse",) else 4)
        rows.append(dict(metric=L(r.metric), scale=f(r["scale"], d), abs=f(r.sd_abs_ctrl, d), dm=f(r.sd_delta_median, d),
                         dx=f(r.sd_delta_max, d), z=f"{f(r.sd_delta_median_z, 3)} / {f(r.sd_delta_max_z, 3)}"))
    return md(pd.DataFrame(rows), ["metric", "scale", "abs", "dm", "dx", "z"],
              ["metric", "column scale s_m", "seed SD of ctrl's value", "seed SD of an improvement over ctrl, median",
               "maximum", "in units of s_m (median / max)"])


def tab_mixed_weights():
    t = T("mixed_weights")
    t = t[t.seed == 0]
    rows = []
    for loss in CFG("mixed"):
        q = t[t.loss == loss]
        rows.append(dict(loss=M.LOSS_LABEL[loss], w=f"{q.weight.mean():.3g}", r=f"{q.weight.min():.3g} - {q.weight.max():.3g}",
                         n=len(q)))
    return md(pd.DataFrame(rows), ["loss", "w", "r", "n"], ["metric loss", "weight w, mean over scenes", "range", "scenes"])


def tab_votes():
    t = T("cross_matrix")
    q = t[t.split == "test"]
    rows = []
    for c in trained():
        g = q[q.cond == c]
        better = g[g.finding & (g.z > 0)]
        worse = g[g.finding & (g.z < 0)]
        rows.append(dict(model=lab(c), nb=len(better), b=", ".join(L(m) for m in better.metric) or "-", nw=len(worse),
                         psnr=fs(X(c, "psnr", "delta")), own=fs(S(c, "own_z"))))
    return md(pd.DataFrame(rows), ["model", "nb", "b", "nw", "psnr", "own"],
              ["model", "metrics that rate it better than ctrl", "which", "metrics that rate it worse", "PSNR change (dB)",
               "z of its own metric"])


def tab_diag():
    t = T("baseline_diagnostics")
    rows = []
    s = "mip360/bicycle"
    rows.append(dict(run="base model of the experiment (gsplat default 0.0002)", N=f"{BV(s, 'num_gaussians') / 1e6:.2f} M",
                     psnr=f(BV(s, "psnr")), ssim=f(BV(s, "ssim"), 3), lp=f(BV(s, "lpips_alex"), 3)))
    for _, r in t.sort_values("grow_grad2d").iterrows():
        rows.append(dict(run=f"densification threshold {r.grow_grad2d:g}", N=f"{r.num_gaussians / 1e6:.2f} M",
                         psnr=f(r.psnr), ssim=f(r.ssim, 3), lp=f(r.lpips_alex, 3)))
    g = "gsplat_docs_gsplat30k"
    rows.append(dict(run="gsplat documentation (commit 6acdce4)", N=f"{f(PUB(g, s, 'num_gaussians_m'))} M", psnr=f(PUB(g, s)),
                     ssim=f(PUB(g, s, "ssim"), 3), lp=f(PUB(g, s, "lpips"), 3)))
    return md(pd.DataFrame(rows), ["run", "N", "psnr", "ssim", "lp"], ["bicycle, stage A", "Gaussians", "PSNR", "SSIM", "LPIPS-Alex"])


def tab_unit():
    c = json.load(open(os.path.join(CTX["R"], "unit_tests.json")))["results"]
    rows = []
    refs = {"psnr": "scikit-image", "ssim": "scikit-image (Gaussian weights, sigma 1.5)", "ms_ssim": "piq",
            "dists": "piq (256x256 crops, no resizing)", "flip": "flip-evaluator (official C++)",
            "cvvdp_image": "pycvvdp, uint8 input", "lpips_vgg": "existing project (lpips package)",
            "lpips_alex": "existing project (lpips package)"}
    for m in M.IMAGE_METRICS:
        if m not in c:
            continue
        key = "reference_" + m if "reference_" + m in c else "reference_" + m + "_vs_existing_project"
        r = c.get(key)
        rows.append(dict(metric=L(m), ident=f(float(np.mean(c[m]["identity"])), 4), grad=f"{c[m]['grad_norm']:.2e}",
                         ref=refs.get(m, "-"), diff=("-" if r is None else f"{r['max_abs_diff']:.1e}")))
    return md(pd.DataFrame(rows), ["metric", "ident", "grad", "ref", "diff"],
              ["metric", "M(x, x)", "gradient norm (finite)", "reference implementation", "max. abs. difference"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("template")
    ap.add_argument("out")
    args = ap.parse_args()
    setup(args.config)
    src = open(args.template).read()
    env = {k: v for k, v in globals().items() if not k.startswith("__")}
    def repl(m):
        return str(eval(m.group(1).strip(), env))
    out = re.sub(r"\{\{(.+?)\}\}", repl, src, flags=re.S)
    open(args.out, "w").write(out)
    print("written", args.out, f"({len(re.findall(r'{{', src))} expressions)")
if __name__ == "__main__":
    main()
