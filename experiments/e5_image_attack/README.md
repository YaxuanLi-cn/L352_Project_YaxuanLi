# E5 图像攻击

从 E4 重跑的模型出发做微调攻击：只守住论文里常用的那几个指标，看能不能把其他指标对应的图像质量推坏。
都在留出视角上测量。

实验分 6 个阶段，按顺序做：

1. `s1_exploration_truck`：在 truck 上试各种攻击损失。
2. `s2_sufficiency_matrix`：算 13 个指标的种子噪声阈值，跑攻击矩阵并判定结果。
3. `s3_tint_truck`：在 truck 上逐步加大 DreamSim 的推力，测相对对照图像的色偏，生成对比页面。
4. `s4_tint_scenes`：在其他场景上重复色偏实验。
5. `s5_detectors`：在被攻击的模型和对照模型上算 13 个指标。
6. `s6_round2`：把 DreamSim 也加进守住的集合，再攻击集合外的指标。

## 运行

要先有 E4 重跑的模型。每个阶段目录里的 `env.sh` 设置运行目录和记录目录，进到阶段目录后先 source 它再跑脚本，例如：

```bash
cd code/experiments/e5_image_attack/s4_tint_scenes
source env.sh
python run_colour.py --scene gsqa/flowers --mult 1   # GPU：攻击一个场景
python colour.py                                     # 算色偏，写 levels.csv
```

各阶段跑攻击的脚本是 `run_ft.py`、`run_suff.py`、`run_tint.py`、`run_colour.py`、`run_ext.py`（要 GPU），
其余 Python 脚本做汇总、判定和画图，`.sh` 文件是当时排队用的。
运行结果在 `data/experiments/e5_image_attack/<阶段>/`，汇总在 `records/e5_image_attack/<阶段>/`。
