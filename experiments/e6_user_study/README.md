# E6 用户实验：网页上的 2AFC

E5 里三个指标（PSNR、SSIM、LPIPS）都守住了、画面却变了的被攻击模型，以及 E3 里逐帧指标没有反应的闪烁视频，
人能不能看出来？参与者在网页上看一张参考图和两张渲染图 A、B，选出更接近参考图的那张。先做图像题，最后几题是视频。
A、B 的位置和题目顺序对每个人随机，另有 2 个检查试次（其中一个选项就是参考图本身），答错的参与者整个剔除。
分析方案在收数据之前写好，见 `records/e6_user_study/ANALYSIS_PLAN.md`。

`build.py`、`build_ext.py` 生成刺激和清单，`server.py` 是网站服务器，`index.html` 是参与者看到的页面，
`analyse.py` 汇总作答。

## 运行

都不需要 GPU。数据目录是 `data/experiments/e6_user_study/study_data/`。

```bash
# 1. 生成刺激（要 E1、E4、E5 的渲染结果，以及 PyTorch 和 ffmpeg）
python build.py        # 图片、视频和 manifest.json
python build_ext.py    # 追加第二轮攻击的试次

# 2. 启动网站
python3 server.py --port 10200

# 3. 分析
python3 analyse.py
```

`server.py` 启动时会打印两个密钥。参与者打开 `http://<host>:<port>/s/<study key>/`，
管理员打开 `http://<host>:<port>/admin/<admin key>/` 看进度和下载作答。
`server.py` 和 `analyse.py` 都可以用 `--data DIR` 换数据目录。
分析结果写在 `records/e6_user_study/results/` 下的 `summary.md` 和 `summary.json`。
