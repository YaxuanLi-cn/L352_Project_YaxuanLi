# E8 自我高估是不是两次训练之间的随机漂移

这也是一次没有成功的尝试。

E4 的每个配置正好训练过两次（原始运行和重跑，阶段 A 模型、种子、损失都相同），但两次渲染出来的图并不一样。
想检验的是：如果换损失造成的损害主要就是这种每次训练都不一样的漂移，那自我高估只是噪声。

结果是随机漂移不是主要原因：损害的大部分在两次训练里都会重现，把两次训练的图平均以后 S 基本不变。

代码分两部分，都只用 E4 已有的渲染图，不训练新模型：

- `drift.py`、`drift_analysis.py`、`drift_figures.py`：在两次训练的渲染图之间直接算 13 个指标，看漂移大小和 S 的关系。
- `decomposition/`：把一次训练的平方误差拆成两次都重现的部分和每次训练特有的部分。

## 运行

要先做完 E4 的原始运行和重跑，`drift_analysis.py` 还要用 E7 的 `tiles_long.parquet`。

```bash
# 两次训练对比
python drift.py pairs        # GPU
python drift_analysis.py
python drift_figures.py

# 分解
cd decomposition
python step1_pixel.py gsqa/truck           # 每个场景跑一次
python step2_metrics.py gsqa/truck 10 abm  # GPU：两次训练的平均图上的指标
python aggregate.py
python make_finding.py                     # 生成 FINDING.md 和图
```

`step2_metrics.py` 的三个参数是场景、最多用几个视角、模式（GS-QA 场景用 `abm`，Mip-NeRF 360 场景用 `m`）。
结果在 `records/e8_drift/` 下的 `two_trainings/` 和 `decomposition/`。
