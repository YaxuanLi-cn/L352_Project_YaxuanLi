#!/usr/bin/env python
import argparse
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import helpers as C              
import uncertainty as U              
import matplotlib              
matplotlib.use("Agg")
import matplotlib.pyplot as plt              
import numpy as np              
import pandas as pd              
from matplotlib.colors import LinearSegmentedColormap              
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIV = LinearSegmentedColormap.from_list("div", ["#b3261e", "#e66767", "#f0efec", "#6da7ec", "#184f95"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#9fc3ef", "#2a78d6", "#123c70"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 200, "figure.dpi": 100})
STRATA = ["Q1", "Q2", "Q3", "Q4", "Q5"]
SPLIT = {"test": "held-out views", "train": "training views"}
GROUP = {"pure": "single-metric losses (9)", "mixed": "default + metric (7)"}


def grid(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def errline(ax, x, d, color, marker, label, offset=0.0):
    x = np.asarray(x, float) + offset
    ax.errorbar(x, d.value, yerr=[d.value - d.lo, d.hi - d.value], color=color, marker=marker, ms=4.5, lw=1.6,
                capsize=2, elinewidth=0.9, label=label, markeredgecolor="white", markeredgewidth=0.6)


def fig_strata(res, out):
    a1 = pd.read_csv(os.path.join(res, "A1_strata_S.csv"))
    a1 = a1[a1.reference == "paper"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for r, split in enumerate(("test", "train")):
        for c, group in enumerate(("pure", "mixed")):
            ax = axes[r, c]
            g = a1[(a1.split == split) & (a1.group == group)]
            ax.axhline(0, color=MUTED, lw=0.7)
            for q, col, mk, lab, off in (("S", BLUE, "o", "self-overrating S", -0.12),
                                         ("own_z", ORANGE, "s", "own metric z", 0.0),
                                         ("ref_z", AQUA, "^", "other-family metrics z", 0.12)):
                d = g[(g.quantity == q) & g.stratum.isin(STRATA)].set_index("stratum").loc[STRATA]
                errline(ax, range(5), d, col, mk, lab, off)
                ov = g[(g.quantity == q) & (g.stratum == "overall")]
                if q == "S" and len(ov):
                    ax.axhline(float(ov.value.iloc[0]), color=BLUE, lw=0.8, ls="--", label="S, all tiles")
            ax.set_xticks(range(5), ["Q1\ncertain", "Q2", "Q3", "Q4", "Q5\nuncertain"])
            ax.set_title(f"{SPLIT[split]}, {GROUP[group]}", color=INK)
            grid(ax)
            if c == 0:
                ax.set_ylabel("z (units of $s_m$)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    order = [lab.index(x) for x in ("self-overrating S", "S, all tiles", "own metric z", "other-family metrics z")]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(os.path.join(out, "fig2_strata_S.png"))
    fig.savefig(os.path.join(out, "fig2_strata_S.pdf"))
    plt.close(fig)


def fig_confirmed(res, out):
    a2 = pd.read_csv(os.path.join(res, "A2_confirmed_share.csv"))
    order = STRATA + ["Q1+Q2"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
    for r, split in enumerate(("test", "train")):
        for c, group in enumerate(("pure", "mixed")):
            ax = axes[r, c]
            g = a2[(a2.split == split) & (a2.group == group)].set_index("stratum").loc[order]
            ax.axhline(0, color=MUTED, lw=0.7)
            ax.axhline(1, color=MUTED, lw=0.7, ls=":")
            x = np.arange(len(order))
            for xi, (s, row) in zip(x, g.iterrows()):
                ok = bool(row.reported)
                ax.errorbar(xi, row.C, yerr=[[row.C - row.lo], [row.hi - row.C]], color=BLUE if s != "Q1+Q2" else ORANGE,
                            marker="o" if s != "Q1+Q2" else "D", ms=5, mfc=(BLUE if s != "Q1+Q2" else ORANGE) if ok else "white",
                            capsize=2, elinewidth=0.9, lw=0)
                if not ok:
                    ax.text(xi, row.hi, "own gain\nn.s.", ha="center", va="bottom", fontsize=6, color=INK2)
            ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4", "Q5", "Q1+Q2"])
            ax.set_title(f"{SPLIT[split]}, {GROUP[group]}")
            grid(ax)
            if c == 0:
                ax.set_ylabel("confirmed share C\n(sum of other-family z / sum of own z)")
    fig.text(0.5, 0.005, "hollow marker: the own-metric improvement in this stratum is not significant (95% CI includes 0); "
             "C is then not interpreted", ha="center", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out, "fig3_confirmed_share.png"))
    fig.savefig(os.path.join(out, "fig3_confirmed_share.pdf"))
    plt.close(fig)


def fig_concentration(res, out):
    curves = pd.read_json(os.path.join(res, "A3_curves.json"))
    top = pd.read_csv(os.path.join(res, "A3_top20.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True, sharey=True)
    for r, split in enumerate(("test", "train")):
        for c, group in enumerate(("pure", "mixed")):
            ax = axes[r, c]
            g = curves[(curves.split == split) & (curves.group == group)]
            x = np.asarray(g.x.iloc[0])
            ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":", label="no concentration")
            for _, row in g.iterrows():
                ax.plot(x, row.S, color=BLUE, lw=0.6, alpha=0.35)
            ax.plot(x, np.mean(np.stack(g.S.to_numpy()), axis=0), color=BLUE, lw=2, label="S (mean of scenes)")
            ax.plot(x, np.mean(np.stack(g.err.to_numpy()), axis=0), color=ORANGE, lw=1.6, ls="--",
                    label="squared error of the control")
            ax.axvline(0.2, color=GRID, lw=0.8)
            t = top[(top.split == split) & (top.group == group)].set_index("quantity")
            if "share_S_top20" in t.index:
                v = t.loc["share_S_top20"]
                ax.text(0.22, 0.04, f"top 20% uncertain tiles: {v.value:.2f} of S\n[{v.lo:.2f}, {v.hi:.2f}]",
                        fontsize=6.5, color=INK2, va="bottom")
            ax.set_title(f"{SPLIT[split]}, {GROUP[group]}")
            if r == 1:
                ax.set_xlabel("share of tiles, most uncertain first")
            if c == 0:
                ax.set_ylabel("cumulative share")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(os.path.join(out, "fig4_concentration.png"))
    fig.savefig(os.path.join(out, "fig4_concentration.pdf"))
    plt.close(fig)


def fig_agreement(res, out):
    a4 = pd.read_csv(os.path.join(res, "A4_agreement.csv"))
    order = STRATA + ["overall"]
    qs = [("kendall_tau", "Kendall $\\tau$ (mean of 21 metric pairs)"),
          ("concordance_all_pairs", "concordance, all model pairs"),
          ("concordance_resolved_pairs", "concordance, pairs beyond seed noise")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, (q, title) in zip(axes, qs):
        ax.axhline(0, color=MUTED, lw=0.7)
        for src, col, mk, lab, off in (("loss", BLUE, "o", "17 loss models", -0.1), ("ladder", ORANGE, "s", "3DGS iteration ladder", 0.1)):
            d = a4[(a4.source == src) & (a4.quantity == q)].set_index("stratum")
            d = d.reindex(order).dropna(subset=["value"])
            if len(d):
                errline(ax, [order.index(s) for s in d.index], d, col, mk, lab, off)
        ax.set_xticks(range(6), ["Q1", "Q2", "Q3", "Q4", "Q5", "all"])
        ax.set_title(title, fontsize=7.5)
        grid(ax)
    axes[0].set_ylabel("agreement of the 7 metrics")
    axes[0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_agreement.png"))
    fig.savefig(os.path.join(out, "fig5_agreement.pdf"))
    plt.close(fig)


def example_views(spec, views):
    n = len(views)
    return [views[int(round(n / 3))], views[int(round(2 * n / 3)) - 1]]


def fig_examples(pilot_out, out):
    import lpips
    import torch
    import tiles as TL
    ctx = TL.Ctx(argparse.Namespace(out=pilot_out, scenes=None, max_views=None, metrics=None, models=None,
                                    allow_missing_seeds=False, loop=False))
    net = lpips.LPIPS(net="vgg", spatial=True, verbose=False).cuda().eval()
    def lp(a, b):
        t = torch.from_numpy(a).cuda().permute(2, 0, 1)[None].float() / 255
        r = torch.from_numpy(b).cuda().permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            return net(t, r, normalize=True)[0, 0].cpu().numpy()
    ready = [s for s in ctx.scenes
             if os.path.exists(os.path.join(ctx.rcfg["views_dir"], f"{C.scene_dir(s)}.json"))
             and os.path.isdir(ctx.png(s, "s0", "lpips_vgg__mixed", "test"))                
             and all(os.path.exists(ctx.png(s, "seeds", str(k), "done")) for k in ctx.seeds(s))]              
    for dataset, scenes in (("gsqa", [s for s in ready if s.startswith("gsqa/")]),
                            ("mip360", [s for s in ready if s.startswith("mip360/")])):
        rows = [(s, v) for s in scenes for v in example_views(s, ctx.views(s)["test"])]
        if not rows:
            continue
        first = C.read_pngs([ctx.png(rows[0][0], "reference", "test", rows[0][1][1])])[0]
        h, w = first.shape[:2]
        fig = plt.figure(figsize=(7.2, 7.2 * len(rows) * h / w / 5 * 1.08 + 0.4))
        gs = fig.add_gridspec(len(rows), 5, left=0.06, right=0.995, top=0.97, bottom=0.03, wspace=0.03, hspace=0.06)
        titles = ["Reference", "Default loss (control)", "Default + LPIPS-VGG", "U: seed SD per pixel",
                  "LPIPS-VGG gain (blue = better)"]
        for ri, (spec, (vid, name)) in enumerate(rows):
            ref = C.read_pngs([ctx.png(spec, "reference", "test", name)])[0]
            ctrl = C.read_pngs([ctx.png(spec, "s0", "default__ctrl", "test", name)])[0]
            vgg = C.read_pngs([ctx.png(spec, "s0", "lpips_vgg__mixed", "test", name)])[0]
            seeds = ctx.seeds(spec)
            sd = np.sqrt(U.variance_map(C.read_pngs([ctx.png(spec, "seeds", str(s), "test", name) for s in seeds])))
            gain = lp(ctrl, ref) - lp(vgg, ref)
            y0, x0, nr, nc = U.tile_grid(h, w, ctx.t)
            panels = [(ref, None), (ctrl, None), (vgg, None), (sd, ("seq", 0, np.percentile(sd, 99))),
                      (gain, ("div", -np.percentile(np.abs(gain), 99), np.percentile(np.abs(gain), 99)))]
            for ci, (img, cm) in enumerate(panels):
                ax = fig.add_subplot(gs[ri, ci])
                if cm is None:
                    ax.imshow(img, interpolation="lanczos")            
                else:
                    ax.imshow(img, cmap=SEQ if cm[0] == "seq" else DIV, vmin=cm[1], vmax=cm[2], interpolation="nearest")             
                if ci >= 3:
                    for k in range(nr + 1):
                        ax.axhline(y0 + k * ctx.t - 0.5, color=INK2, lw=0.35)
                    for k in range(nc + 1):
                        ax.axvline(x0 + k * ctx.t - 0.5, color=INK2, lw=0.35)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if ri == 0:
                    ax.set_title(titles[ci], fontsize=7)
                if ci == 0:
                    ax.set_ylabel(f"{spec.split('/')[1]} #{vid}", fontsize=7, color=INK2)
        fig.text(0.5, 0.005, f"U map: per-pixel SD of the renders of {len(seeds)} default-loss models with other seeds "
                 f"(colour scale 0 to the 99th percentile of the view); lines: {ctx.t}-pixel tiles", ha="center",
                 fontsize=6.5, color=INK2)
        fig.savefig(os.path.join(out, f"fig1_examples_{dataset}.png"))
        plt.close(fig)
FIGS = dict(strata=fig_strata, confirmed=fig_confirmed, concentration=fig_concentration, agreement=fig_agreement)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=P.PILOT)
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    res = os.path.join(args.out, "results")
    fdir = os.path.join(args.out, "figures")
    os.makedirs(fdir, exist_ok=True)
    for name in args.names or ["examples", *FIGS]:
        if name == "examples":
            fig_examples(args.out, fdir)
        else:
            FIGS[name](res, fdir)
        C.log(f"figure {name} written to {fdir}")
if __name__ == "__main__":
    main()
