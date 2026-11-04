# E1 退化阶梯

在 8 个场景上（Mip-NeRF 360 的 kitchen、bonsai、garden、bicycle，GS-QA 的 truck、train、playground2、flowers）
用 3DGS 和 NeRF 训练模型，沿三条阶梯把模型弄差：迭代数、训练视角数、高斯数上限（只有 3DGS）。
每个场景 30 个条件，一共 240 个模型。GS-QA 的 4 个场景另外换一个随机种子重训了一部分，用来估计种子噪声。
然后在所有模型上算指标，看各个指标给出的排序是否一致，以及逐帧算的指标漏掉了多少闪烁。

训练、渲染和指标的实现在 `code/common/nvsqa/`。

## 运行

先用 `code/common/download/` 里的脚本把 Mip-NeRF 360 和 GS-QA 下载到 `data/datasets/`。

```bash
SCENES="gsqa/truck gsqa/train mip360/kitchen mip360/bonsai mip360/garden mip360/bicycle gsqa/playground2 gsqa/flowers"

# 1. 训练并渲染（GPU），两个队列可以同时跑
python run_queue.py --scenes $SCENES --methods 3dgs &
python run_queue.py --scenes $SCENES --methods nerf &

# 2. 第二个随机种子（GPU），只在 GS-QA 的 4 个场景上
SEEDS="3dgs-full@1 3dgs-views-4@1 3dgs-views-16@1 3dgs-gauss-30000@1 3dgs-gauss-300000@1 nerf-full@1 nerf-views-4@1 nerf-views-16@1"
python run_queue.py --scenes gsqa/truck gsqa/train gsqa/playground2 gsqa/flowers --jobs $SEEDS

# 3. 算指标（GPU），加 --loop 可以和训练同时开着
python eval_part1.py --loop

# 4. 汇总和分析，等前面全部做完再跑
python collect_part1.py      # conditions.csv、frames.csv
python describe_part1.py     # GPU，descriptors.csv
python analyse_part1.py      # 其余的表
```

已经做完的任务会跳过，中断后重新运行同一条命令就行。
渲染结果在 `data/experiments/e1_degradation_ladders/`，结果表在 `records/e1_degradation_ladders/results/`。
