# E2 Agreement between metrics and observer scores

We compute the metrics on two public datasets with subjective scores (GS-QA: 64 Gaussian splatting videos; NVS-QA: 88 NeRF
videos) and look at how well each metric correlates with the human scores (DMOS).
`unanimity/` goes one step further: when two methods are compared on the same scene, how reliable is the conclusion when
PSNR, SSIM and LPIPS all point the same way, and when they do not.

The metrics are implemented in `code/common/nvsqa/`.

## How to run

Download both datasets with `code/common/download/download_gsqa.py` (add `--repo NVS-QA` for NVS-QA) into `data/datasets/`.
Step 3 also needs the result tables of E1 and E4.

```bash
# 1. compute the metrics on every video (GPU)
python eval_subjective.py --dataset GS-QA --extra stlpips_alex stlpips_vgg
python eval_subjective.py --dataset NVS-QA

# 2. correlation with DMOS and confidence intervals
python analyse_subjective.py --dataset gsqa
python analyse_subjective.py --dataset nvsqa

# 3. do the three metrics agree
cd unanimity
python c2_unanimity.py             # main analysis, run this first
python human_ceiling.py            # how reliable a single observer is (NVS-QA only)
python bridge_by_cause.py          # the same rule applied to the E1 and E4 models
python published_values_check.py   # redo with the metric values published by the dataset authors
python tied_pairs.py               # video pairs where the metrics tie: how different are they to people
python make_finding.py             # writes FINDING.md
```

Results are written to `gsqa/`, `nvsqa/` and `unanimity/` under `records/e2_observer_agreement/`.
