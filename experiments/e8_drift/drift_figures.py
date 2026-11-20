#!/usr/bin/env python
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
sys.path.insert(0, P.code(P.E7))
import helpers as C              
import figures as F              
import matplotlib.pyplot as plt              
import numpy as np              
import pandas as pd              
from matplotlib.colors import LogNorm              
from gh import metrics as M              
OUT = P.DRIFT
RES, FIG = os.path.join(OUT, "results"), os.path.join(OUT, "figures")
KCOL = {"pixel": F.MUTED, "structural": F.AQUA, "perceptual": F.BLUE}
LOSS_LABEL = dict(M.LOSS_LABEL, lpips_squeeze="LPIPS-Squeeze", charbonnier="Charbonnier", dreamsim="DreamSim", vif="VIF", haarpsi="HaarPSI")


def name(model):
    loss, var = model.split("__")[:2]
    return LOSS_LABEL[loss] if var == "pure" else f"default + {LOSS_LABEL[loss]}"


def save(fig, stem):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def d1():
    t = pd.read_csv(os.path.join(RES, "blind_ratio.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, col, title in ((axes[0], "ratio", "in s.d. units of the paper ($s_m$)"),
                           (axes[1], "ratio_rel_ctrl", "relative to the control model's drift")):
        y = np.arange(len(t))[::-1]
        for yi, (_, r) in zip(y, t.iterrows()):
            ax.errorbar(r[col], yi, xerr=[[r[col] - r[col + "_lo"]], [r[col + "_hi"] - r[col]]], color=KCOL[r.klass],
                        marker="o" if r.group == "pure" else "s", ms=4.5, capsize=2, elinewidth=0.9, markeredgecolor="white", markeredgewidth=0.6)
        ax.axvline(1, color=F.INK2, lw=0.8, ls="--")
        ax.set_xscale("log")
        ticks = [t_ for t_ in (0.05, 0.1, 0.2, 0.5, 1, 2, 5) if t[col + "_lo"].min() * 0.8 <= t_ <= t[col + "_hi"].max() * 1.25]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t_:g}" for t_ in ticks])
        ax.minorticks_off()
        ax.set_yticks(y)
        ax.set_yticklabels([name(m) for m in t.model])
        ax.set_xlabel("own-metric drift / other-metric drift")
        ax.set_title(title)
        ax.grid(axis="x", color=F.GRID, lw=0.5)
        ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=c, marker="o", ls="", ms=4.5, label=k) for k, c in KCOL.items()]
    h += [plt.Line2D([], [], color=F.INK2, marker=mk, ls="", ms=4.5, label=l) for mk, l in (("o", "metric alone"), ("s", "default + metric"))]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Drift between two trainings: what the loss's own metric sees, relative to the other metrics", y=1.0, fontsize=9)
    save(fig, "figd1_blind_ratio")


def d2():
    t = pd.read_csv(os.path.join(RES, "per_loss.csv"))
    fits = json.load(open(os.path.join(RES, "results.json")))["fit_S_on_log10_drift"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, (ycol, ylab) in zip(axes, (("S", "self-overrating $S$ (mean of the two trainings)"), ("magnitude_other", "change against the control (other metrics, s.d.)"))):
        for _, r in t.iterrows():
            ax.plot(r.drift_other, r[ycol], color=KCOL[r.klass], marker="o" if r.group == "pure" else "s", ms=5, ls="",
                    markeredgecolor="white", markeredgewidth=0.6)
            ax.annotate(name(r.model), (r.drift_other, r[ycol]), xytext=(3, 3), textcoords="offset points", fontsize=5.5, color=F.INK2)
        ax.set_xscale("log")
        ax.set_xlabel("drift between the two trainings (other metrics, s.d.)")
        ax.set_ylabel(ylab)
        F.grid(ax)
        if ycol == "S":
            xs = np.geomspace(t.drift_other.min(), t.drift_other.max(), 50)
            for key, ls in (("pure", "-"), ("all", ":")):
                f = fits[key]
                ax.plot(xs, f["intercept"] + f["slope_per_decade"] * np.log10(xs), color=F.INK2, lw=0.9, ls=ls,
                        label=f"least squares, {'9 single-metric' if key == 'pure' else 'all 16'} losses")
            ax.legend(loc="upper left")
        else:
            ax.set_yscale("log")
    save(fig, "figd2_drift_vs_S")


def d3():
    t = pd.read_csv(os.path.join(RES, "spatial_strata.csv"))
    bys = list(dict.fromkeys(t.strata_by))
    cols = [F.MUTED, F.BLUE, F.ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, g in zip(axes, ("pure", "mixed")):
        for k, by in enumerate(bys):
            d = t[(t.strata_by == by) & (t.group == g) & (t.quantity == "S") & t.stratum.isin(F.STRATA)]
            F.errline(ax, range(5), d.reset_index(drop=True), cols[k], "os^"[k], by, offset=(k - 1) * 0.08)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"])
        ax.set_title(F.GROUP[g])
        ax.set_xlabel("stratum of the tiles")
        F.grid(ax)
    axes[0].set_ylabel("self-overrating $S$ (held-out views)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.04), title="tiles stratified by")
    save(fig, "figd3_spatial_strata")


def d4():
    t = pd.read_csv(os.path.join(RES, "drift_by_model_metric.csv"))
    p = t.pivot(index="model", columns="metric", values="drift_sd")
    models = list(dict.fromkeys(t.model))
    mets = [m for m in M.IMAGE_METRICS if m != "psnr"]
    a = p.loc[models, mets].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(a, cmap=F.SEQ, norm=LogNorm(vmin=max(a.min(), 1e-3), vmax=a.max()), aspect="auto")
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels([M.LABEL[m] for m in mets], rotation=40, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(["default (control)" if m.startswith("default") else name(m) for m in models])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j, i, f"{a[i, j]:.2f}" if a[i, j] < 10 else f"{a[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                    color="white" if a[i, j] > np.sqrt(a.min() * a.max()) * 3 else F.INK)
    for i, m in enumerate(models):
        if m.startswith("default"):
            continue
        j = mets.index(M.LOSSES[m.split("__")[0]])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=F.ORANGE, lw=1.3))
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="difference between the two trainings (s.d. units)")
    ax.set_title("Difference between two trainings of the same configuration, per metric (orange: the loss's own metric)")
    save(fig, "figd4_drift_matrix")


def main():
    for f in (d1, d2, d3, d4):
        f()
    C.log("figures written to " + FIG)
if __name__ == "__main__":
    main()
