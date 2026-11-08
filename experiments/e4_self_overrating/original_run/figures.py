#!/usr/bin/env python
import argparse
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import matplotlib              
matplotlib.use("Agg")
import matplotlib.pyplot as plt              
import numpy as np              
import pandas as pd              
import yaml              
from matplotlib.colors import LinearSegmentedColormap              
from matplotlib.patches import Rectangle              
from gh import metrics as M              
DIV = LinearSegmentedColormap.from_list("div", ["#9b2f2e", "#e34948", "#f0efec", "#3987e5", "#104281"])
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#d9d8d4"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white"})
IMG = M.IMAGE_METRICS


def heat(ax, mat, rows, cols, vmax, title, stars=None, fmt="{:.1f}", boxes=()):
    im = ax.imshow(np.clip(mat, -vmax * 3, vmax * 3), cmap=DIV, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=50, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title, loc="left", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                txt = fmt.format(v) + ("*" if stars is not None and stars[i, j] else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.6 * vmax else INK)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i, j in boxes:
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor=INK, linewidth=1.2))
    return im


def own_boxes(conds):
    out = []
    for i, c in enumerate(conds):
        loss, var = c.split("/")
        if var in ("pure", "mixed"):
            out.append((i, IMG.index(M.LOSSES[loss])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = os.path.normpath(os.path.join(P.ROOT, cfg["results_dir"]))
    T = os.path.join(R, "tables")
    F = os.path.join(R, "figures")
    os.makedirs(F, exist_ok=True)
    cols = [M.LABEL[m] for m in IMG]
    cm = pd.read_csv(os.path.join(T, "cross_matrix.csv"))
    conds = list(dict.fromkeys(cm.cond))
    labels = [cm[cm.cond == c].label.iloc[0] for c in conds]
    for sp in ("test", "train"):
        q = cm[cm.split == sp]
        mat = q.pivot(index="cond", columns="metric", values="z").loc[conds, IMG].to_numpy()
        st = q.pivot(index="cond", columns="metric", values="finding").loc[conds, IMG].to_numpy().astype(bool)
        fig, ax = plt.subplots(figsize=(8.2, 0.33 * len(conds) + 1.6))
        im = heat(ax, mat, labels, cols, 3.0,
                  f"{'Held-out (test)' if sp == 'test' else 'Training'} views: improvement over ctrl, column-standardised",
                  stars=st, boxes=own_boxes(conds))
        ax.set_ylabel("training loss (rows)")
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label("z  (> 0: better than ctrl)")
        fig.text(0.01, -0.02, "Box: the metric the model was trained on.  *: CI excludes 0 and |z| > 2 x seed SD.",
                 fontsize=6.5, color=MUTED)
        fig.savefig(os.path.join(F, f"cross_matrix_{sp}.png"))
        plt.close(fig)
    so = pd.read_csv(os.path.join(T, "self_overestimation.csv"))
    so = so[so.reference == "metrics"]
    cl = list(dict.fromkeys(so.cond))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.3 * len(cl) + 1.4), sharey=True)
    for ax, sp in zip(axes, ("test", "train")):
        q = so[so.split == sp]
        y = np.arange(len(cl))[::-1]
        ax.axvline(0, color=MUTED, lw=0.8)
        for yi, c in zip(y, cl):
            s = q[(q.cond == c) & (q.kind == "self")].iloc[0]
            ax.plot([s.lo, s.hi], [yi + 0.14] * 2, color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(s.score, yi + 0.14, "o", color=BLUE, ms=5, mec="white", mew=0.8)
            for k, (_, t) in enumerate(q[(q.cond == c) & (q.kind == "same_family")].iterrows()):
                yy = yi - 0.12 - 0.2 * k
                ax.plot([t.lo, t.hi], [yy] * 2, color=ORANGE, lw=1.6, solid_capstyle="round")
                ax.plot(t.score, yy, "s", color=ORANGE, ms=4, mec="white", mew=0.8)
                ax.text(max(t.hi, t.score) + 0.06, yy, M.LABEL[t.metric], fontsize=5.5, va="center", color=MUTED)
        ax.set_yticks(y, [q[q.cond == c].label.iloc[0] for c in cl])
        ax.set_xlabel("score = z(metric) - mean z(metrics of other families)")
        ax.set_title(f"{'Held-out (test)' if sp == 'test' else 'Training'} views", loc="left", fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].plot([], [], "o", color=BLUE, label="metric used as the loss (self score)")
    axes[0].plot([], [], "s", color=ORANGE, label="other metric of the same family")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    fig.savefig(os.path.join(F, "self_overestimation.png"))
    plt.close(fig)
    om = pd.read_csv(os.path.join(T, "overrating_matrix.csv"))
    q = om[om.split == "test"]
    oc = list(dict.fromkeys(q.cond))
    mat = q.pivot(index="cond", columns="metric", values="score").loc[oc, IMG].to_numpy()
    st = q.pivot(index="cond", columns="metric", values="finding").loc[oc, IMG].to_numpy().astype(bool)
    fig, ax = plt.subplots(figsize=(8.2, 0.33 * len(oc) + 1.6))
    im = heat(ax, mat, [q[q.cond == c].label.iloc[0] for c in oc], cols, 3.0,
              "Held-out views: z(metric) - mean z(metrics outside the families of loss and metric)", stars=st,
              boxes=own_boxes(oc))
    ax.set_ylabel("training loss (rows)")
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).set_label("> 0: rates the model higher than the unrelated metrics do")
    fig.savefig(os.path.join(F, "overrating_matrix.png"))
    plt.close(fig)
    gp = pd.read_csv(os.path.join(T, "generalisation_gap.csv"))
    gc = list(dict.fromkeys(gp.cond))
    glab = [gp[gp.cond == c].label.iloc[0] for c in gc]
    mat = gp.pivot(index="cond", columns="metric", values="gap").loc[gc, IMG].to_numpy()
    st = gp.pivot(index="cond", columns="metric", values="finding").loc[gc, IMG].to_numpy().astype(bool)
    fig, ax = plt.subplots(figsize=(8.2, 0.33 * len(gc) + 1.6))
    im = heat(ax, mat, glab, cols, 3.0, "Generalisation gap: z(training views) - z(held-out views)", stars=st,
              boxes=own_boxes(gc))
    ax.set_ylabel("training loss (rows)")
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).set_label("> 0: gain over ctrl is larger on the training views")
    fig.savefig(os.path.join(F, "generalisation_gap.png"))
    plt.close(fig)
    own = gp[gp.own]
    fig, ax = plt.subplots(figsize=(6.2, 0.3 * len(own) + 1.1))
    y = np.arange(len(own))[::-1]
    ax.axvline(0, color=MUTED, lw=0.8)
    for yi, (_, r) in zip(y, own.iterrows()):
        ax.plot([r.z_test, r.z_train], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.plot(own.z_train, y, "o", color=ORANGE, ms=6, mec="white", mew=0.8, label="training views")
    ax.plot(own.z_test, y, "o", color=BLUE, ms=6, mec="white", mew=0.8, label="held-out views")
    ax.set_yticks(y, own.label)
    ax.set_xlabel("improvement of the trained metric over ctrl (z)")
    ax.set_title("Gain in the metric used as loss", loc="left", fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(os.path.join(F, "own_metric_train_test.png"))
    plt.close(fig)
    vf = pd.read_csv(os.path.join(T, "video_vs_frames.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    lim = [min(vf.d_cvvdp_frames.min(), vf.d_cvvdp_video.min()) * 1.1, max(vf.d_cvvdp_frames.max(), vf.d_cvvdp_video.max()) * 1.6]
    ax.plot(lim, lim, color=GRID, lw=1, zorder=0)
    ax.text(lim[0] * 0.55, lim[0] * 0.62, "video = frames", fontsize=6.5, color=MUTED, rotation=30)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    for _, r in vf.iterrows():
        col = BLUE if "pure" in r.cond else ORANGE if "mixed" in r.cond else MUTED
        ax.plot([r.d_cvvdp_frames] * 2, [r.d_cvvdp_video_lo, r.d_cvvdp_video_hi], color=col, lw=1, alpha=.6)
        ax.plot(r.d_cvvdp_frames, r.d_cvvdp_video, "o", color=col, ms=6, mec="white", mew=1)
        if abs(r.d_cvvdp_video) > 0.1:
            ax.annotate(r.label, (r.d_cvvdp_frames, r.d_cvvdp_video), fontsize=6, xytext=(5, -3), textcoords="offset points")
    ax.set_xlabel("change of ColorVideoVDP, mean of per-frame predictions (JOD) vs ctrl")
    ax.set_ylabel("change of ColorVideoVDP, video mode (JOD) vs ctrl")
    ax.set_title("Video mode against the same metric per frame", loc="left", fontsize=9)
    ax = axes[1]
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    q = vf[vf.own_z.notna()]
    for _, r in q.iterrows():
        col = BLUE if "pure" in r.cond else ORANGE
        ax.plot([r.own_z_lo, r.own_z_hi], [r.d_cvvdp_video] * 2, color=col, lw=1, alpha=.6)
        ax.plot([r.own_z] * 2, [r.d_cvvdp_video_lo, r.d_cvvdp_video_hi], color=col, lw=1, alpha=.6)
        ax.plot(r.own_z, r.d_cvvdp_video, "o", color=col, ms=6, mec="white", mew=1)
        if abs(r.d_cvvdp_video) > 0.1 or r.own_z > 1:
            ax.annotate(r.label, (r.own_z, r.d_cvvdp_video), fontsize=6, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("improvement of the trained metric on the test frames (z) vs ctrl")
    ax.set_ylabel("change of ColorVideoVDP, video mode (JOD) vs ctrl")
    ax.set_title("Video mode against the metric used as loss", loc="left", fontsize=9)
    ax.plot([], [], "o", color=BLUE, label="pure loss")
    ax.plot([], [], "o", color=ORANGE, label="mixed loss")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.savefig(os.path.join(F, "video_vs_frames.png"))
    plt.close(fig)
    tau = pd.read_csv(os.path.join(T, "kendall_tau.csv"))
    common = list(dict.fromkeys(tau.m1))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw=dict(wspace=0.06))
    for ax, nm in zip(axes, ("natural", "finetune")):
        q = tau[tau.set == nm]
        mat = q.pivot(index="m1", columns="m2", values="tau").loc[common, common].to_numpy()
        im = heat(ax, mat, [M.LABEL[m] for m in common] if nm == "natural" else [""] * len(common),
                  [M.LABEL[m] for m in common], 1.0,
                  ("Natural degradations (existing project)" if nm == "natural" else "Loss fine-tuning (this experiment)")
                  + f"\nKendall tau in scene; {q.n_scenes.iloc[0]} scenes, {q.n_conditions.iloc[0]} conditions each",
                  fmt="{:.2f}")
    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01).set_label("Kendall tau")
    fig.savefig(os.path.join(F, "kendall_tau.png"))
    plt.close(fig)
    cc = pd.read_csv(os.path.join(T, "concordance_by_cause.csv"))
    cc = cc[cc.m1.isna()]
    names = {"same_ladder": "same degradation ladder\n(fewer views / iterations / Gaussians)",
             "cross_ladder": "different ladders,\nsame method", "cross_method": "3DGS vs NeRF",
             "finetune": "different training losses\n(this experiment)"}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    order = ["same_ladder", "cross_ladder", "cross_method", "finetune"]
    for i, c in enumerate(order):
        for matched, col, dx in ((False, MUTED, -0.12), (True, BLUE, 0.12)):
            if c == "finetune" and not matched:
                continue
            r = cc[(cc.pair_class == c) & (cc.range_matched == matched)].iloc[0]
            x = i + (0 if c == "finetune" else dx)
            ax.plot([x, x], [r.lo, r.hi], color=col, lw=2, solid_capstyle="round")
            ax.plot(x, r.concordance, "o", color=col, ms=7, mec="white", mew=1)
            ax.text(x + 0.07, r.concordance, f"{r.concordance:.2f}", fontsize=7, va="center")
    ax.plot([], [], "o", color=MUTED, label="all pairs of conditions")
    ax.plot([], [], "o", color=BLUE, label="pairs no further apart than the fine-tuned models")
    ax.set_xticks(range(len(order)), [names[c] for c in order], fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("concordance of two metrics\n(mean of 21 metric pairs)")
    ax.set_title("Agreement between metrics by what makes the two models differ", loc="left", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.savefig(os.path.join(F, "concordance_by_cause.png"))
    plt.close(fig)
    print("figures written to", F)
if __name__ == "__main__":
    main()
