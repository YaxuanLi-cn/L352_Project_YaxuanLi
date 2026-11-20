#!/usr/bin/env python
import os
import sys
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
REC = P.DECOMPOSITION
RES = os.path.join(REC, "results")
CTRL = "default__ctrl"
NAME = {"default__ctrl": "对照", "l1__pure": "L1", "l2__pure": "L2", "dssim__pure": "D-SSIM", "msssim__pure": "MS-SSIM",
        "flip__pure": "FLIP", "cvvdp__pure": "ColorVideoVDP", "lpips_vgg__pure": "LPIPS-VGG", "lpips_alex__pure": "LPIPS-Alex",
        "dists__pure": "DISTS", "l2__mixed": "L2（混合）", "msssim__mixed": "MS-SSIM（混合）", "flip__mixed": "FLIP（混合）",
        "cvvdp__mixed": "ColorVideoVDP（混合）", "lpips_vgg__mixed": "LPIPS-VGG（混合）", "lpips_alex__mixed": "LPIPS-Alex（混合）",
        "dists__mixed": "DISTS（混合）"}
DEEP_PURE = ["lpips_vgg__pure", "lpips_alex__pure", "dists__pure"]
DEEP_MIXED = ["lpips_vgg__mixed", "lpips_alex__mixed", "dists__mixed"]


def ci(r, c, d=2):
    return f"{r[c]:.{d}f} [{r[c + '_lo']:.{d}f}, {r[c + '_hi']:.{d}f}]"


def main():
    per = pd.read_csv(os.path.join(RES, "decomposition_per_loss.csv")).set_index("model")
    ps = pd.read_csv(os.path.join(RES, "decomposition_per_scene.csv"))
    ns = ps.scene.nunique()
    single = ps.groupby("model").psnr_single.mean()
    L = []
    L.append("# C1 分解：损失造成的损害有多少在两次训练之间不可复现")
    L.append("")
    L.append("数据：每个场景、每个配置都训练过两次（A = 论文模型，B = 重跑模型；同一 checkpoint、同一种子、不同机器），参考图 G。"
             "只用现有渲染图，没有新的训练。脚本：`step1_pixel.py`（像素域 + 定位）、`step2_metrics.py`（平均图的 13 个指标）、"
             "`aggregate.py`（汇总，生成 `results/summary.md`）、`make_finding.py`（本文件）。本文件的数字全部由脚本从 `results/*.csv` 填入。")
    L.append("")
    L.append("模型：A = μ + a，B = μ + b，a、b 是两次训练各自的部分（独立、同方差 v）。于是 v = mse(A,B)/2 是单次训练误差里"
             "不可复现的部分，bias² = 单次 MSE − v 是可复现的部分；损失相对对照多出来的误差里，不可复现份额 = (v_损失 − v_对照)/(MSE_损失 − MSE_对照)。")
    L.append("")
    L.append("## 关键数字（脚本生成）")
    L.append("")
    L.append(f"- 两次训练彼此之间的 PSNR（{ns} 个场景平均，方括号为按场景自助法 95% 区间）："
             + "；".join(f"{NAME[m]} {ci(per.loc[m], 'psnr_between_runs', 1)} dB" for m in [CTRL, "l1__pure", "cvvdp__pure"] + DEEP_PURE) + "。")
    L.append(f"- 对比：单次训练到参考图的 PSNR：" + "；".join(f"{NAME[m]} {single[m]:.1f} dB" for m in [CTRL] + DEEP_PURE) + "。")
    L.append("- 两次训练之间的 MSE ÷ 单次训练到参考图的 MSE（场景平均）："
             + "；".join(f"{NAME[m]} {2 * per.loc[m, 'v_share_of_error']:.3f}" for m in [CTRL, "l1__pure", "cvvdp__pure"] + DEEP_PURE + DEEP_MIXED) + "。")
    bt = ps.pivot_table(index="scene", columns="model", values="psnr_between_runs")
    n_order = int(((bt[CTRL] > bt["lpips_vgg__pure"]) & (bt["lpips_vgg__pure"] > bt["lpips_alex__pure"]) & (bt["lpips_alex__pure"] > bt["dists__pure"])).sum())
    n_pix = int((bt[["l1__pure", "l2__pure", "dssim__pure", "msssim__pure"]].min(axis=1) > bt[DEEP_PURE].max(axis=1)).sum())
    L.append(f"- 两次训练之间的 PSNR 的顺序“对照 > LPIPS-VGG > LPIPS-Alex > DISTS”在 {n_order}/{ns} 个场景成立；"
             f"“四种像素/结构类单独损失都高于三种深度特征单独损失”在 {n_pix}/{ns} 个场景成立。")
    L.append("- 损失相对对照多出来的 MSE 中不可复现的份额："
             + "；".join(f"{NAME[m]} {ci(per.loc[m], 'irreproducible_share')}（{int(per.loc[m, 'n_scenes_share_above_half'])}/{ns} 个场景超过一半）"
                        for m in DEEP_PURE + DEEP_MIXED if per.loc[m, "excess_reliable"]) + "。")
    small = [m for m in DEEP_PURE + DEEP_MIXED if not per.loc[m, "excess_reliable"]]
    if small:
        L.append("- MSE 增量过小或符号不定、不计算份额的深度特征模型：" + "、".join(NAME[m] for m in small) + "。")
    L.append("- 其他单独损失的不可复现份额："
             + "；".join(f"{NAME[m]} {ci(per.loc[m], 'irreproducible_share')}" for m in ["dssim__pure", "msssim__pure", "flip__pure", "cvvdp__pure"]
                        if per.loc[m, "excess_reliable"]) + "。")
    L.append("- 相对对照的 PSNR 变化（dB）：单次训练 → 两次训练的平均图 → 只留可复现部分（外推到无穷多次训练的平均）："
             + "；".join(f"{NAME[m]} {per.loc[m, 'dpsnr_single']:+.2f} → {per.loc[m, 'dpsnr_mean_image']:+.2f} → {per.loc[m, 'dpsnr_bias_only']:+.2f}"
                        for m in DEEP_PURE + DEEP_MIXED + ["cvvdp__pure"]) + "。")
    f2 = os.path.join(RES, "mean_image_S.csv")
    s2 = pd.read_csv(f2).set_index("model")
    bs = pd.read_csv(os.path.join(RES, "mean_image_S_by_scene.csv"))
    L.append(f"- 两次训练的平均图（{bs.scene.nunique()} 个场景：{'、'.join(sorted(bs.scene.unique()))}）上，自身指标的提升 z_own：单次 → 平均图："
             + "；".join(f"{NAME[m]} {s2.loc[m, 'z_own_single']:+.2f} → {s2.loc[m, 'z_own_mean_image']:+.2f}" for m in DEEP_PURE + DEEP_MIXED)
             + "。其他族指标的平均 z_ref：" + "；".join(f"{NAME[m]} {s2.loc[m, 'z_ref_single']:+.2f} → {s2.loc[m, 'z_ref_mean_image']:+.2f}"
                                               for m in DEEP_PURE + DEEP_MIXED)
             + "。S：" + "；".join(f"{NAME[m]} {s2.loc[m, 'S_single']:+.2f} → {s2.loc[m, 'S_mean_image']:+.2f}" for m in DEEP_PURE + DEEP_MIXED) + "。")
    f3 = os.path.join(RES, "localisation_per_loss.csv")
    lo = pd.read_csv(f3).set_index("model")
    L.append("- 两次训练之差 |A−B| 与“超出对照的误差”在 16×16 像素块上的 Pearson 相关："
             + "；".join(f"{NAME[m]} {ci(lo.loc[m], 'pearson_excess')}" for m in DEEP_PURE + DEEP_MIXED + ["cvvdp__pure", "flip__pure"]) + "。")
    L.append("")
    L.append(open(os.path.join(REC, "INTERPRETATION.md")).read().rstrip())
    L.append("")
    L.append("---")
    L.append("")
    L.append("# 全部表格（`results/summary.md`，脚本生成）")
    L.append("")
    L.append(open(os.path.join(RES, "summary.md")).read().rstrip())
    L.append("")
    L.append("文件：`results/decomposition_per_loss.csv`、`decomposition_per_scene.csv`、`localisation_per_loss.csv`、`localisation_per_scene.csv`、"
             "`mean_image_S.csv`、`mean_image_S_by_scene.csv`、`mean_image_z_all_metrics.csv`、`fig_decomposition.png`；逐视角原始值在 `out/`。")
    open(os.path.join(REC, "FINDING.md"), "w").write("\n".join(L) + "\n")
    order = ["l1__pure", "l2__pure", "dssim__pure", "msssim__pure", "flip__pure", "cvvdp__pure", "lpips_vgg__pure", "lpips_alex__pure",
             "dists__pure", "l2__mixed", "msssim__mixed", "flip__mixed", "cvvdp__mixed", "lpips_vgg__mixed", "lpips_alex__mixed", "dists__mixed"]
    g = ps.groupby("model")[["excess_v_rel", "excess_bias_rel"]].mean().loc[order]
    lab = [m.replace("__pure", "").replace("__mixed", "\n(mixed)") for m in order]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(order))
    ax[0].bar(x, g.excess_bias_rel, color="#4c72b0", label="reproducible (in both trainings)")
    ax[0].bar(x, g.excess_v_rel, bottom=np.where(g.excess_bias_rel > 0, g.excess_bias_rel, 0), color="#dd8452",
              label="run-specific (differs between the two trainings)")
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(lab, rotation=60, ha="right", fontsize=8)
    ax[0].set_ylabel("excess MSE over the control / control MSE")
    ax[0].set_title(f"Excess squared error of loss-trained models, {ns} scenes", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    allm = [CTRL] + order
    b = per.loc[allm]
    ax[1].errorbar(np.arange(len(allm)), b.psnr_between_runs, yerr=[b.psnr_between_runs - b.psnr_between_runs_lo, b.psnr_between_runs_hi - b.psnr_between_runs],
                   fmt="o", color="#dd8452", label="between the two trainings")
    ax[1].plot(np.arange(len(allm)), single.loc[allm], "s", color="#4c72b0", label="one training vs the reference")
    ax[1].set_xticks(np.arange(len(allm)))
    ax[1].set_xticklabels(["ctrl"] + lab, rotation=60, ha="right", fontsize=8)
    ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_title("How far apart are two trainings of the same configuration?", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_decomposition.png"), dpi=140)
if __name__ == "__main__":
    main()
