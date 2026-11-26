# Sources of published_baselines.csv

The values in this table are not results of this project. They were copied on 2026-11-20/21 from the places below and are
used for comparison with our base models.

- `kerbl2023`: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", 2023,
  appendix tables 4–6 ("Ours-30k"), read from https://ar5iv.labs.arxiv.org/html/2308.04079 .
  LPIPS is computed by the official code, which feeds [0,1] images to the VGG network.
- `gsplat_docs_inria30k`, `gsplat_docs_gsplat30k`: the evaluation page of gsplat,
  https://docs.gsplat.studio/main/tests/eval.html ("Reproduced Metrics", the official Inria code at commit 36546ce and
  gsplat at commit 6acdce4). LPIPS is the AlexNet version from torchmetrics (normalize=True).
- `rocm_gsplat153_30k`: AMD's reproduction with gsplat 1.5.3b2,
  https://rocm.docs.amd.com/projects/gsplat/en/latest/reference/benchmark-evaluation.html
  (two decimals only, the LPIPS network is not stated). The data loader of this version re-creates the down-sampled images
  as PNG with bicubic resizing, so the images it compares against are not exactly the same as ours.
