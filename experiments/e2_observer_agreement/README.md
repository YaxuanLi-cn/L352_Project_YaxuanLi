# E2 指标和观察者评分的一致性

在两个公开的主观评分数据集上（GS-QA：64 个高斯泼溅视频；NVS-QA：88 个 NeRF 视频）算各个指标，
看它们和人打的分（DMOS）相关到什么程度。
`unanimity/` 里进一步看：比较同一场景的两个方法时，PSNR、SSIM、LPIPS 三个指标方向一致和不一致的情况下，
结论各有多可靠。

指标的实现在 `code/common/nvsqa/`。

## 运行

两个数据集用 `code/common/download/download_gsqa.py` 下载（NVS-QA 加 `--repo NVS-QA`），放在 `data/datasets/` 下。
第 3 步还要用到 E1 和 E4 的结果表。

```bash
# 1. 在每个视频上算指标（GPU）
python eval_subjective.py --dataset GS-QA --extra stlpips_alex stlpips_vgg
python eval_subjective.py --dataset NVS-QA

# 2. 指标和 DMOS 的相关、置信区间
python analyse_subjective.py --dataset gsqa
python analyse_subjective.py --dataset nvsqa

# 3. 三个指标是否一致的分析
cd unanimity
python c2_unanimity.py             # 主分析，先跑这个
python human_ceiling.py            # 单个观察者有多可靠（只有 NVS-QA）
python bridge_by_cause.py          # 同一条规则用到 E1、E4 的模型上
python published_values_check.py   # 换成数据集作者发布的指标值重算
python tied_pairs.py               # 指标打平的视频对，人眼里差多少
python make_finding.py             # 生成 FINDING.md
```

结果写在 `records/e2_observer_agreement/` 下的 `gsqa/`、`nvsqa/` 和 `unanimity/`。
