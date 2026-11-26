# E6 User study: 2AFC in the browser

In E5 some attacked models keep all three metrics (PSNR, SSIM, LPIPS) but the picture has changed, and in E3 the flickering
videos do not move the per-frame metrics. Can people see the difference? A participant sees a reference image and two renders
A and B on a web page, and picks the one that is closer to the reference. Image trials come first, the last few trials are videos.
The position of A and B and the order of the trials are random for each participant. There are 2 check trials where one of
the options is the reference image itself, and a participant who gets one wrong is removed completely.
The analysis plan was written before collecting data, see `records/e6_user_study/ANALYSIS_PLAN.md`.

`build.py` and `build_ext.py` generate the stimuli and the manifest, `server.py` is the web server, `index.html` is the page
the participants see, and `analyse.py` summarises the responses.

## How to run

No GPU is needed. The data directory is `data/experiments/e6_user_study/study_data/`.

```bash
# 1. generate the stimuli (needs the renders of E1, E4 and E5, plus PyTorch and ffmpeg)
python build.py        # images, videos and manifest.json
python build_ext.py    # add the trials of the second attack round

# 2. start the website
python3 server.py --port 10200

# 3. analysis
python3 analyse.py
```

`server.py` prints two keys when it starts. Participants open `http://<host>:<port>/s/<study key>/`,
the admin opens `http://<host>:<port>/admin/<admin key>/` to see the progress and download the responses.
Both `server.py` and `analyse.py` take `--data DIR` to use another data directory.
The results are `summary.md` and `summary.json` in `records/e6_user_study/results/`.
