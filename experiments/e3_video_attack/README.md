# E3 视频攻击：往视频里注入闪烁

E1 里同一个退化条件训练了两个只有随机种子不同的模型，它们的相机路径视频记作 A 和 B。
每 k 帧在 A、B 之间切换一次，拼成混合视频（k = 1、2、4、8、15）。混合视频的每一帧都原样来自 A 或 B，
所以逐帧指标的平均不变，但看起来会闪。这个实验看 ColorVideoVDP 的视频模式能不能看出这种闪烁。
另外把 A 和 B 逐像素取平均（`avg`）作为对照。

## 运行

要先跑完 E1（两个种子的视频都在）。

```bash
python temporal_probe.py      # GPU：构造混合视频，算 ColorVideoVDP
python analyse_probes.py      # 汇总
```

`temporal_probe.py` 可以用 `--scenes`、`--conditions` 只跑一部分，中断后直接重跑。
结果在 `records/e3_video_attack/results/` 下的 `temporal_probe.csv` 和 `temporal_probe_summary.csv`。
