# E4 自我高估：把指标当作 3DGS 的训练损失

把某个质量指标直接当作损失去微调一个训练好的 3DGS，看这个指标会不会把自己训练出来的模型评得比别的指标高。

- 阶段 A：每个场景用默认损失（0.8 L1 + 0.2 D-SSIM）训练一个 3DGS，30000 步。8 个场景，truck 做 3 个种子。
- 阶段 B：从阶段 A 的模型出发，高斯集合固定，换一种损失再训练 10000 步，然后在留出视角和训练视角上算 13 个指标。

每个（场景, 种子）有 18 个运行：不微调的 `base`、仍用默认损失的对照 `ctrl`、9 个只用单个指标当损失的 `pure`、
7 个默认损失加指标损失的 `mixed`，一共 180 个运行。
自我高估 S = 模型在自己的指标上相对 `ctrl` 的提升，减去它在其他指标上的平均提升，定义见 `original_run/analyse.py`。

实验做了两次。`original_run/` 是第一次，当时没有保存微调后的模型和逐视角的渲染图；
`rerun/` 用同样的阶段 A 模型、种子和权重重做了阶段 B，并把渲染图和模型都存了下来。
后面的实验（E5–E8）用的都是重跑的结果。

训练和指标的实现在 `code/common/nvsqa/gs.py` 和 `code/common/gh/`，
其中 `gh/flip_torch.py` 拷贝自 https://github.com/NVlabs/flip 。

## 运行

原始运行：

```bash
cd original_run
python run.py configs/full.yaml                          # GPU：阶段 A + 180 个阶段 B 的运行
python baseline_check.py conventions configs/full.yaml   # GPU：按官方 3DGS 的算法重算 base 模型的指标
python extra_rows.py configs/full.yaml
python stimuli_manifest.py configs/full.yaml
python analyse.py configs/full.yaml                      # 结果表
python figures.py configs/full.yaml                      # 图
python figure_examples.py configs/full.yaml --scene mip360/garden
```

重跑（要有阶段 A 的模型和原始运行的 `runs/`）：

```bash
cd rerun
python rerun_finetune.py configs/rerun.yaml   # GPU：180 个运行
python recompute_whole.py                     # 用 ../original_run/analyse.py 重算结果表
python compare_whole.py                       # 和原始运行的表对比
python compare_renders.py                     # 和原始运行的渲染图对比
```

`run.py` 和 `rerun_finetune.py` 都可以续跑，也可以同时开几个进程，加 `--list` 只看每个运行的状态。
模型和渲染图在 `data/experiments/e4_self_overrating/`，结果表在 `records/e4_self_overrating/`。
