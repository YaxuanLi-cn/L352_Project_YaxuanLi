# published_baselines.csv 的数据来源

表里的数值不是本项目的结果，是 2026-11-20/21 从下面几处抄来的，用来和我们的 base 模型对比。

- `kerbl2023`：Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", 2023，
  附录表 4–6（"Ours-30k"），取自 https://ar5iv.labs.arxiv.org/html/2308.04079 。
  LPIPS 由官方代码计算，输入 VGG 网络的图像范围是 [0,1]。
- `gsplat_docs_inria30k`、`gsplat_docs_gsplat30k`：gsplat 的评测页面
  https://docs.gsplat.studio/main/tests/eval.html （"Reproduced Metrics"，Inria 官方代码 commit 36546ce 和
  gsplat commit 6acdce4）。LPIPS 是 torchmetrics 的 AlexNet 版本（normalize=True）。
- `rocm_gsplat153_30k`：AMD 用 gsplat 1.5.3b2 复现的结果，
  https://rocm.docs.amd.com/projects/gsplat/en/latest/reference/benchmark-evaluation.html
  （只有两位小数，没有写 LPIPS 用的网络）。这个版本的数据加载会把降采样图像用双三次插值重新生成成 PNG，
  所以对比的图像和我们的不完全一样。
