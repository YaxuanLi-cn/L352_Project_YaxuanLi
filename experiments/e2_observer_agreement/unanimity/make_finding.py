#!/usr/bin/env python
import os
import sys
import pandas as pd
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
OUT = P.UNANIMITY
R = lambda f: pd.read_csv(os.path.join(OUT, f))
single, votes, levels, cons, boot, tie, chk, strat = (R(f + ".csv") for f in (
    "single_metric_accuracy", "vote_accuracy", "vote_levels", "consensus", "bootstrap_contrasts", "tie_rule", "check_against_existing", "stratified_by_ddmos"))
VN = {"triplet_vgg": "PSNR / SSIM / LPIPS-VGG", "triplet_alex": "PSNR / SSIM / LPIPS-Alex", "five_family": "PSNR / SSIM / LPIPS-Alex / FLIP / CVVDP"}
SN = {"all": "全部", "real": "真实场景", "synthetic": "合成场景", "FF": "前向(FF)", "360": "360°", "FF-synthetic": "前向合成", "360-synthetic": "360°合成"}


def md(rows, header):
    out = ["| " + " | ".join(h.replace("|", "\\|") for h in header) + " |", "|" + "|".join([":--"] + ["--:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


def f2(x):
    return "–" if pd.isna(x) else f"{x:.2f}"


def bget(ds, sub, pairs, vote, q):
    r = boot[(boot.dataset == ds) & (boot.subset == sub) & (boot.pairs == pairs) & (boot.vote == vote) & (boot.quantity == q)]
    return r.iloc[0] if len(r) else None


def cistr(r, signed=True):
    if r is None:
        return "–"
    s = "+" if signed and r.value >= 0 else ""
    return f"{s}{r.value:.2f} [{r.lo:+.2f}, {r.hi:+.2f}]"
L = []
L.append("# C2：跨族指标\"一致同意\"规则与零参数共识分数（GS-QA、NVS-QA 主观数据）\n")
L.append("脚本：`c2_unanimity.py`（分析）、`make_finding.py`（本文件，所有数字从 CSV 填入）。只用现有结果表，不用 GPU。\n")
L.append("## 做法\n")
L.append("- 数据：`records/e2_observer_agreement/gsqa/metrics.csv`（64 个视频，8 个场景 × 8 种方法）和 `records/e2_observer_agreement/nvsqa/metrics.csv`（88 个视频，16 个场景）。DMOS 越高越好"
         "（两张表都已核对）。\n- 只看**场景内的方法对**（同一场景、两种方法）：GS-QA "
         f"{int(votes[(votes.dataset=='GS-QA')&(votes.subset=='all')&(votes.pairs=='all pairs')].n_pairs.iloc[0])} 对，NVS-QA "
         f"{int(votes[(votes.dataset=='NVS-QA')&(votes.subset=='all')&(votes.pairs=='all pairs')].n_pairs.iloc[0])} 对。人的判断 = DMOS 差的符号。\n"
         "- \"一致同意\"= 一组指标对这一对的判断方向全部相同；\"分歧\"= 至少一个指标方向相反，此时取多数票。\n"
         "- 置信区间：按**整个场景**重采样 2000 次（GS-QA 8 个场景，NVS-QA 16 个场景）。\n")
c = chk
L.append("## 实现核对\n")
L.append(f"单指标的成对准确率和场景内 Spearman 与项目已有的 `agreement_all.csv` 逐项一致（{len(c)} 行，最大绝对差 "
         f"{max((c.acc_all_here - c.acc_all_existing).abs().max(), (c.acc_d05_here - c.acc_d05_existing).abs().max(), (c.within_srocc_here - c.within_srocc_existing).abs().max()):.1e}）。\n")
L.append("## 结果 1：一致同意规则在两个数据集上都成立\n")
rows = []
for ds in ("GS-QA", "NVS-QA"):
    for pairs in ("all pairs", "|dDMOS|>0.5"):
        for v in VN:
            r = votes[(votes.dataset == ds) & (votes.subset == "all") & (votes.pairs == pairs) & (votes.vote == v)].iloc[0]
            rows.append([ds, "全部对" if pairs == "all pairs" else "|ΔDMOS|>0.5", VN[v], int(r.n_pairs), f"{100*r.share_unanimous:.0f}%",
                         f"{r.acc_unanimous:.2f} ({int(r.n_unanimous)})", f"{r.acc_split_majority:.2f} ({int(r.n_split)})",
                         cistr(bget(ds, "all", pairs, v, "gap"))])
L.append(md(rows, ["数据集", "方法对", "投票的指标", "对数", "一致的比例", "一致时与人相符（对数）", "分歧时多数票与人相符（对数）", "差值 [95% CI，按场景重采样]"]))
L.append("\n读法：以论文通用的 PSNR / SSIM / LPIPS-VGG 为例，约四分之三的方法对三个指标方向一致；一致时与观察者相符的比例明显高于分歧时，"
         "分歧时多数票接近抛硬币。\n")
L.append("### 按场景类别\n")
rows = []
for ds, subs in (("GS-QA", ["FF", "360"]), ("NVS-QA", ["synthetic", "real", "FF-synthetic", "360-synthetic", "FF", "360"])):
    for sub in subs:
        r = votes[(votes.dataset == ds) & (votes.subset == sub) & (votes.pairs == "all pairs") & (votes.vote == "triplet_vgg")]
        r = r.iloc[0]
        rows.append([ds, SN[sub], int(r.n_scenes), int(r.n_pairs), f"{100*r.share_unanimous:.0f}%", f"{r.acc_unanimous:.2f} ({int(r.n_unanimous)})",
                     f"{r.acc_split_majority:.2f} ({int(r.n_split)})", cistr(bget(ds, sub, "all pairs", "triplet_vgg", "gap"))])
L.append(md(rows, ["数据集", "场景类别", "场景数", "对数", "一致的比例", "一致时相符（对数）", "分歧时多数票相符（对数）", "差值 [95% CI]"]))
L.append("\n（PSNR / SSIM / LPIPS-VGG，全部方法对。NVS-QA 的真实 360° 场景渲染与参考有 3–6 像素错位：三个指标只在少数方法对上一致，"
         "每个子类只有 4 个场景，区间很宽。）\n")
L.append("### 五族投票：多数的大小与准确率\n")
rows = []
for ds in ("GS-QA", "NVS-QA"):
    for _, r in levels[(levels.dataset == ds) & (levels.subset == "all") & (levels.pairs == "all pairs") & (levels.vote == "five_family")].sort_values("majority_size", ascending=False).iterrows():
        rows.append([ds, r.majority_size, int(r.n), f"{100*r.share:.0f}%", f"{r.acc_majority:.2f}"])
L.append(md(rows, ["数据集", "票型", "对数", "占比", "多数票与人相符"]))
L.append("\n")
L.append("### 一致性是否只是\"差别大的对更容易\"？按 |ΔDMOS| 分层\n")
rows = []
for ds in ("GS-QA", "NVS-QA"):
    for _, r in strat[(strat.dataset == ds) & (strat.subset == "all") & (strat.vote == "triplet_vgg")].iterrows():
        rows.append([ds, r.ddmos_bin, int(r.n), f"{100*r.share_unanimous:.0f}%", f"{f2(r.acc_unanimous)} ({int(r.n_unanimous)})", f"{f2(r.acc_split_majority)} ({int(r.n_split)})"])
L.append(md(rows, ["数据集", "|ΔDMOS| 区间", "对数", "一致的比例", "一致时相符（对数）", "分歧时多数票相符（对数）"]))
L.append("\n（PSNR / SSIM / LPIPS-VGG。同一 |ΔDMOS| 区间内比较一致与分歧。）\n")
L.append("## 结果 2：指标分歧时该信谁\n")
rows = []
for ds in ("GS-QA", "NVS-QA"):
    for sub in (["all"] if ds == "GS-QA" else ["all", "synthetic", "real"]):
        for v in ("triplet_vgg", "triplet_alex"):
            a, b_, c_ = (bget(ds, sub, "all pairs", v, q) for q in ("split_acc_lpips", "split_acc_psnr", "split_acc_ssim"))
            d1, d2 = bget(ds, sub, "all pairs", v, "split_lpips_minus_psnr"), bget(ds, sub, "all pairs", v, "split_lpips_minus_ssim")
            n = int(votes[(votes.dataset == ds) & (votes.subset == sub) & (votes.pairs == "all pairs") & (votes.vote == v)].n_split.iloc[0])
            rows.append([ds, SN[sub], VN[v], n, f"{a.value:.2f}", f"{b_.value:.2f}", f"{c_.value:.2f}", cistr(d1), cistr(d2)])
L.append(md(rows, ["数据集", "子集", "三元组", "分歧的对数", "LPIPS 相符", "PSNR 相符", "SSIM 相符", "LPIPS − PSNR [95% CI]", "LPIPS − SSIM [95% CI]"]))
r5 = votes[(votes.subset == "all") & (votes.pairs == "all pairs") & (votes.vote == "five_family")].set_index("dataset")
L.append("\n五族投票分歧的对上各指标与人相符的比例：" + "；".join(
    f"{ds}（{int(r5.loc[ds].n_split)} 对）LPIPS-Alex {r5.loc[ds].acc_split_lpips_alex:.2f}、CVVDP {r5.loc[ds].acc_split_cvvdp:.2f}、"
    f"PSNR {r5.loc[ds].acc_split_psnr:.2f}、SSIM {r5.loc[ds].acc_split_ssim:.2f}、FLIP {r5.loc[ds].acc_split_flip:.2f}" for ds in ("GS-QA", "NVS-QA")) + "。\n")
L.append("## 结果 3：零参数共识分数**没有**超过最好的单指标\n")
rows = []
for ds in ("GS-QA", "NVS-QA"):
    s = single[(single.dataset == ds) & (single.subset == "all")]
    def g(metric, pairs, col="pair_acc"):
        return s[(s.metric == metric) & (s.pairs == pairs)][col].iloc[0]
    for m in ("lpips_alex", "cvvdp", "ssim", "psnr", "lpips_vgg", "flip"):
        rows.append([ds, s[s.metric == m].label.iloc[0], f"{g(m,'all pairs'):.3f}", f"{g(m,'|dDMOS|>0.5'):.3f}", f"{g(m,'all pairs','within_srocc'):.3f}", ""])
    cc = cons[(cons.dataset == ds) & (cons.subset == "all")]
    for v in VN:
        for kind, kn in (("cons_z_", "z 均值"), ("cons_r_", "名次均值")):
            m = kind + v
            a = cc[(cc.metric == m) & (cc.pairs == "all pairs")].iloc[0]
            b_ = cc[(cc.metric == m) & (cc.pairs == "|dDMOS|>0.5")].iloc[0]
            rows.append([ds, f"共识（{kn}；{VN[v]}）", f"{a.pair_acc:.3f}", f"{b_.pair_acc:.3f}", f"{a.within_srocc:.3f}", cistr(bget(ds, "all", "all pairs", v, m + "_minus_lpips_alex"))])
L.append(md(rows, ["数据集", "指标", "成对准确率（全部对）", "成对准确率（|ΔDMOS|>0.5）", "场景内 Spearman", "共识 − LPIPS-Alex 的成对准确率 [95% CI]"]))
L.append("\n共识分数与多数票基本相同，比 PSNR 略好或持平，但在两个数据集上都低于 LPIPS-Alex。所以\"把几族指标平均成一个新指标\"不是创新点；"
         "有用的是\"一致 / 分歧\"这个标志本身。\n")
L.append("## 结果 4（附加）：小于种子噪声的差别不算数\n")
import json
thr = json.load(open(os.path.join(OUT, "results.json")))["seed_noise_p90"]
L.append("门槛 = 本项目 `records/e1_degradation_ladders/results/seed_noise.csv` 中只差随机种子的两次训练之差的 90% 分位：" +
         "、".join(f"{k} {v:.3g}" for k, v in thr.items() if k in ("psnr", "ssim", "lpips_alex", "lpips_vgg", "flip", "cvvdp")) + "。\n")
rows = []
for ds in ("GS-QA", "NVS-QA"):
    t = tie[(tie.dataset == ds) & (tie.subset == "all") & (tie.pairs == "all pairs")]
    for m in ("psnr", "ssim", "lpips_vgg", "lpips_alex", "cvvdp", "vote:triplet_vgg", "vote:five_family"):
        r = t[t.metric == m].iloc[0]
        name = {"vote:triplet_vgg": "三元组规则：≥2 个指标超过门槛且方向全同", "vote:five_family": "五族规则：≥2 个指标超过门槛且方向全同"}.get(m, m)
        rows.append([ds, name, f"{100*r.coverage:.0f}%", f"{r.acc_decided:.2f}", f"{f2(r.acc_undecided)}", f"{r.acc_all:.2f}"])
L.append(md(rows, ["数据集", "指标 / 规则", "给出判断的比例", "给出判断时相符", "不判断的对上（若强行判断）相符", "全部对上相符"]))
L.append("\n（规则两行的第 5 列是\"超过门槛的指标之间互相矛盾\"的那些对上多数票的相符比例。门槛来自我们自己训练的模型，"
         "用在别人训练的 GS-QA / NVS-QA 方法上只是近似。）\n")
g1, n1 = bget("GS-QA", "all", "all pairs", "triplet_vgg", "gap"), bget("NVS-QA", "all", "all pairs", "triplet_vgg", "gap")
vg = votes[(votes.subset == "all") & (votes.pairs == "all pairs") & (votes.vote == "triplet_vgg")].set_index("dataset")
vb = votes[(votes.subset == "all") & (votes.pairs == "|dDMOS|>0.5") & (votes.vote == "triplet_vgg")].set_index("dataset")
L.append("## 结论\n")
L.append(f"- **一致同意规则成立（两个数据集）。** PSNR / SSIM / LPIPS-VGG 一致时与观察者相符 {vg.loc['GS-QA'].acc_unanimous:.2f}（GS-QA）/ "
         f"{vg.loc['NVS-QA'].acc_unanimous:.2f}（NVS-QA）；分歧时多数票只有 {vg.loc['GS-QA'].acc_split_majority:.2f} / {vg.loc['NVS-QA'].acc_split_majority:.2f}。"
         f"差值 {cistr(g1)} 与 {cistr(n1)}，按场景重采样的区间都不含 0。人眼差别明确（|ΔDMOS|>0.5）的对上：一致时 "
         f"{vb.loc['GS-QA'].acc_unanimous:.2f} / {vb.loc['NVS-QA'].acc_unanimous:.2f}，分歧时 {vb.loc['GS-QA'].acc_split_majority:.2f} / {vb.loc['NVS-QA'].acc_split_majority:.2f}"
         f"（GS-QA 这一格只有 {int(vb.loc['GS-QA'].n_split)} 对分歧，差值 {cistr(bget('GS-QA','all','|dDMOS|>0.5','triplet_vgg','gap'))}，区间含 0）。\n"
         "- **分歧时多数票不可信；LPIPS-Alex 是分歧对上最常与人相符的单个指标**（见结果 2），在有错位的真实场景上尤其明显；但在位姿精确的合成场景上各指标差不多，"
         "区间都宽，不能写成一般规律。三元组里用 LPIPS-VGG 时，GS-QA 的分歧对上三个指标都接近 0.5，没有哪个可以单独信。\n"
         "- **共识分数不成立为创新点**：不如 LPIPS-Alex。\n"
         "- 可写进论文的形式：一条评测规则——报告三个指标是否方向一致；一致且差别超过种子噪声时可以下结论，分歧时不要数票，应看图或做主观实验。\n")
L.append("## 局限\n")
L.append("- 场景数少（8 和 16），所有区间都是按场景重采样得到的，子类（每类 4 个场景）的区间很宽。\n"
         "- 两个数据集里的方法都用常规损失训练；这条规则没有在\"用指标当损失训练出的模型\"上验证过（那里指标的分歧正是论文 §6 的主题，需要我们自己的主观实验）。\n"
         "- \"一致\"的对平均而言质量差别也更大；分层表显示同一 |ΔDMOS| 区间内差距多数仍在，但各格样本少。\n"
         "- NVS-QA 公开视频是有损压缩版本，指标是在其上计算的。\n")
open(os.path.join(OUT, "FINDING.md"), "w").write("\n".join(L))
print("\n".join(L))
