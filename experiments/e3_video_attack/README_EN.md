# E3 Video attack: injecting flicker into a video

In E1 the same degradation condition was trained twice with different random seeds. Call the two camera-path videos A and B.
We switch between A and B every k frames to build a mixed video (k = 1, 2, 4, 8, 15). Every frame of the mixed video is taken
unchanged from A or B, so the average of any per-frame metric stays the same, but the video flickers. This experiment checks
whether the video mode of ColorVideoVDP can see this flicker.
As a control we also average A and B pixel by pixel (`avg`).

## How to run

E1 has to be finished first (videos of both seeds are needed).

```bash
python temporal_probe.py      # GPU: build the mixed videos and compute ColorVideoVDP
python analyse_probes.py      # summary
```

`temporal_probe.py` takes `--scenes` and `--conditions` to run only a part, and can simply be restarted after an interruption.
Results are `temporal_probe.csv` and `temporal_probe_summary.csv` in `records/e3_video_attack/results/`.
