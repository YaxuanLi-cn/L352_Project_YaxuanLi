# E7 自我高估是不是集中在场景不确定的区域

这是一次没有成功的尝试。

想检验的假设是：场景里没有被数据确定下来的地方，每个指标各选各的折中，所以 E4 的自我高估 S 应该集中在这些地方。

1. 每个场景另外训练 4 个默认损失的 3DGS，和 E4 的基准模型只差随机种子。这 4 个模型的渲染图逐像素算方差，作为不确定性图。
2. 图像切成 192×192 的块，每个场景内按块的不确定性分成 5 层（Q1 最确定，Q5 最不确定）。
3. 对 E4 重跑的每个微调模型，在每个块上算 13 个指标，然后在每一层里重算 S。

结果是不支持这个假设：S 从 Q1 到 Q5 只升了一点，最不确定的 20% 的块只占 S 的 22%。
详细结果见 `records/e7_uncertainty_strata/`。

## 运行

要先有 E4 阶段 A 的模型和 E1 的 3DGS 迭代数阶梯。

```bash
bash run_all.sh              # 全部步骤（包含 E4 的重跑）
bash run_all.sh analysis     # 跳过训练、渲染和 E4 的重跑，只做分块之后的步骤
```

`run_all.sh` 依次执行的是：

```bash
python -m pytest -q tests ../../common/tests/test_uncertainty.py
python train_seeds.py configs/pilot.yaml      # GPU：训练种子模型
python render_seeds.py configs/pilot.yaml ../e4_self_overrating/rerun/configs/rerun.yaml   # GPU
python tiles.py strata       # 分层
python tiles.py checks       # GPU：核对指标实现
python tiles.py metrics      # GPU：分块指标
python tiles.py ladder       # GPU：E1 阶梯上的分块指标
python tiles.py pixel        # GPU：像素级版本
python tiles.py table        # 拼成长表
python analysis.py           # 分析和判定
python figures.py            # 画图
```

每一步都会跳过已经有的结果，中断后直接重跑。`launch_4gpu/` 是当时在 4 块 GPU 上排任务用的脚本，单机复现用不到。
种子模型在 `data/experiments/e7_uncertainty_strata/ckpt_seeds/`，结果在 `records/e7_uncertainty_strata/pilot/`。
