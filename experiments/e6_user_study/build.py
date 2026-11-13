#!/usr/bin/env python
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
ROOT = P.ROOT
RERUN = P.RERUN_STIMULI
OUT = P.STUDY_DATA
SECRET = "nvs-suff-2afc-2026-11-25"
SCENES = ["gsqa/truck", "gsqa/flowers", "gsqa/playground2", "gsqa/train"]
ATTACK = {"gsqa/truck": os.path.join(P.S2_RUNS, "gsqa_truck", "front-triplet-dreamsim-0.0125__from-ctrl__3000__s0"),
          "gsqa/flowers": os.path.join(P.S4_RUNS, "gsqa_flowers", "triplet-dreamsim-aligned-x1__from-ctrl__3000__s0"),
          "gsqa/playground2": os.path.join(P.S4_RUNS, "gsqa_playground2", "triplet-dreamsim-aligned-x0.5__from-ctrl__3000__s0"),
          "gsqa/train": os.path.join(P.S4_RUNS, "gsqa_train", "triplet-dreamsim-aligned-x0.5__from-ctrl__3000__s0")}
VIEW_IDX = [0, 4, 8, 12, 16]
PERIOD, SECONDS, CRF = 4, 6.0, 17


def held_views(sdn):
    v = json.load(open(os.path.join(P.RERUN_VIEWS, f"{sdn}.json")))
    h = np.asarray(v["heldout"])
    return [int(x) for x in h[np.unique(np.round(np.linspace(0, len(h) - 1, 20)).astype(int))]]


def opaque(*parts, ext):
    return hashlib.sha1("|".join([SECRET] + [str(p) for p in parts]).encode()).hexdigest()[:14] + ext


def save_jpg(src, name):
    Image.open(src).convert("RGB").save(os.path.join(OUT, "img", name), "JPEG", quality=97, subsampling=0, optimize=True)


def encode(frames, fps, path):
    f, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-an",
           "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF), "-profile:v", "high", "-g", str(int(round(fps))),
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", path]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def main():
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "vid"), exist_ok=True)
    trials, files = [], {}
    for s in SCENES:
        sdn = s.replace("/", "_")
        hv = held_views(sdn)
        src = {"reference": os.path.join(RERUN, sdn, "reference", "test"), "control": os.path.join(RERUN, sdn, "s0", "default__ctrl", "test"),
               "attacked": os.path.join(ATTACK[s], "test"), "floor": os.path.join(RERUN, sdn, "s0", "lpips_vgg__mixed", "test")}
        def img(cond, v):
            name = opaque("img", sdn, cond, v, ext=".jpg")
            if name not in files:
                save_jpg(os.path.join(src[cond], f"{v:03d}.png"), name)
                files[name] = dict(scene=s, cond=cond, view=v)
            return name
        for i in VIEW_IDX:
            v = hv[i]
            trials.append(dict(tid=f"attack/{sdn}/{v}", part="image", kind="attack", scene=s, view=v, ref=img("reference", v),
                               a=img("control", v), b=img("attacked", v), a_cond="control", b_cond="attacked"))
        v = hv[10]
        trials.append(dict(tid=f"floor/{sdn}/{v}", part="image", kind="floor", scene=s, view=v, ref=img("reference", v),
                           a=img("control", v), b=img("floor", v), a_cond="control", b_cond="floor"))
    for s, i in (("gsqa/truck", 2), ("gsqa/flowers", 14)):
        sdn = s.replace("/", "_")
        v = held_views(sdn)[i]
        d = os.path.join(RERUN, sdn)
        names = {}
        for cond, p in (("reference", os.path.join(d, "reference", "test")), ("control", os.path.join(d, "s0", "default__ctrl", "test"))):
            names[cond] = opaque("img", sdn, cond, v, ext=".jpg")
            if names[cond] not in files:
                save_jpg(os.path.join(p, f"{v:03d}.png"), names[cond])
                files[names[cond]] = dict(scene=s, cond=cond, view=v)
        dup = opaque("img", sdn, "reference-copy", v, ext=".jpg")
        save_jpg(os.path.join(d, "reference", "test", f"{v:03d}.png"), dup)
        files[dup] = dict(scene=s, cond="reference-copy", view=v)
        trials.append(dict(tid=f"check/{sdn}/{v}", part="image", kind="check", scene=s, view=v, ref=names["reference"],
                           a=dup, b=names["control"], a_cond="reference-copy", b_cond="control"))
    from nvsqa import pipeline, video_io
    old = json.load(open(os.path.join(HERE, "video_windows.json")))
    for s in SCENES:
        sdn = s.replace("/", "_")
        o = old[sdn]
        ys, xs = slice(o["y"], o["y"] + o["height"]), slice(o["x"], o["x"] + o["width"])
        fps = float(o["fps"])
        n = int(round(SECONDS * fps))
        ref = np.stack([np.asarray(Image.open(os.path.join(RERUN, sdn, "reference", "test", f"{k:03d}.png")).convert("RGB")) for k in range(n)])
        def vid(cond, frames):
            name = opaque("vid", sdn, cond, ext=".mp4")
            encode(frames[:, ys, xs], fps, os.path.join(OUT, "vid", name))
            files[name] = dict(scene=s, cond=cond, fps=fps, frames=int(frames.shape[0]), y=o["y"], x=o["x"])
            return name
        r = vid("reference", ref)
        for lv, kind in (("views-1of16", "flicker"), ("iters-30000", "vfloor")):
            a = video_io.read_video(os.path.join(pipeline.PART1_ROOT, s, "3dgs", lv, "video.mp4"))[0][:n]
            b = video_io.read_video(os.path.join(pipeline.PART1_ROOT, s, "3dgs", lv + "-s1", "video.mp4"))[0][:n]
            mix = a.copy()
            sel = (np.arange(n) // PERIOD) % 2 == 1
            mix[sel] = b[sel]
            na, nb, nm = vid(lv, a), vid(lv + "-s1", b), vid(f"mix{PERIOD}:" + lv, mix)
            for parent, pn in ((lv, na), (lv + "-s1", nb)):
                trials.append(dict(tid=f"{kind}/{sdn}/{parent}", part="video", kind=kind, scene=s, ref=r, a=pn, b=nm, a_cond=parent,
                                   b_cond=f"mix{PERIOD}:" + lv, group=f"{kind}/{sdn}"))
        print(s, "videos done", fps, n)
    man = dict(version=1, period=PERIOD, seconds=SECONDS, crf=CRF, jpeg_quality=97, trials=trials, files=files)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    isz = sum(os.path.getsize(os.path.join(OUT, "img", f)) for f in os.listdir(os.path.join(OUT, "img"))) / 1e6
    vsz = sum(os.path.getsize(os.path.join(OUT, "vid", f)) for f in os.listdir(os.path.join(OUT, "vid"))) / 1e6
    print(len(trials), "trials;", f"images {isz:.1f} MB, videos {vsz:.1f} MB")
if __name__ == "__main__":
    main()
