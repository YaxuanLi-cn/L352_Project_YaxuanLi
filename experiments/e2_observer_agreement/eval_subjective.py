#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys
import time
os.environ.setdefault("OMP_NUM_THREADS", "4")
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import numpy as np              
import openpyxl              
import pandas as pd              
from nvsqa import DATA_ROOT, descriptors, video_io              
from nvsqa.metrics import IMAGE_METRICS, VIDEO_METRICS, MetricSuite              
GSQA_METHODS = {
    "3dgs": "3DGS", "eagles": "EAGLES", "lightgaussian": "LightGaussian", "mipsplatting": "Mip-Splatting",
    "octree": "Octree-GS", "scaffold": "Scaffold-GS", "sog_sh": "SOG", "sog": "SOG (w/o SH)",
}
GSQA_STLPIPS_SWAP = {"antique": "m60", "flowers": "playground", "playground2": "train", "statue": "truck"}
SCENE_CLASS = {"m60": "360", "playground": "360", "train": "360", "truck": "360",
               "antique": "FF", "flowers": "FF", "playground2": "FF", "statue": "FF",
               "drums": "360-synthetic", "ficus": "360-synthetic", "lego": "360-synthetic", "ship": "360-synthetic",
               "classroom": "FF-synthetic", "mugs": "FF-synthetic", "office": "FF-synthetic", "tea": "FF-synthetic"}
NVSQA_METHODS = {
    "dvgo": "DVGO", "dvgo_ss2": "DVGO (w/ ss.)", "instantNGP": "Instant-NGP", "instantNGP_ss2": "Instant-NGP (w/ ss.)",
    "plenoxels": "Plenoxels", "plenoxels_ss2": "Plenoxels (w/ ss.)", "tensorf": "TensoRF",
    "tensorf_ss2": "TensoRF (w/ ss.)", "multinerf": "Mip-NeRF 360", "nerfacto": "Nerfacto", "nerfplusplus": "NeRF++",
}


def read_sheet(path):
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() for h in rows[0] if h is not None]
    out, scene = [], None
    for r in rows[1:]:
        r = r[:len(header)]
        scene = r[0] if r[0] is not None else scene
        out.append([str(scene).strip().lower(), str(r[1]).strip(), *r[2:]])
    return pd.DataFrame(out, columns=["scene", "method"] + header[2:])


def gsqa_stimuli(root):
    dmos = read_sheet(os.path.join(root, "dmos.xlsx")).rename(columns={"DMOS": "dmos"})
    pub = read_sheet(os.path.join(root, "metrics.xlsx"))
    swap = dict(GSQA_STLPIPS_SWAP, **{v: k for k, v in GSQA_STLPIPS_SWAP.items()})
    by_key = {(sc, me): v for sc, me, v in zip(pub.scene, pub.method, pub["ST-LPIPS"])}
    pub["ST-LPIPS (sheet order)"] = pub["ST-LPIPS"]
    pub["ST-LPIPS"] = [by_key[(swap[sc], me)] for sc, me in zip(pub.scene, pub.method)]
    pub = pub.rename(columns={c: "pub_" + c for c in pub.columns if c not in ("scene", "method")})
    table = dmos.merge(pub, on=["scene", "method"], how="left")
    stimuli = []
    for path in sorted(glob.glob(os.path.join(root, "Synthesized Videos", "*", "*", "*.mp4"))):
        scene = os.path.basename(os.path.dirname(path))
        suffix = os.path.basename(path)[len(scene) + 1:-4]
        stimuli.append(dict(scene=scene, method=GSQA_METHODS[suffix], test=path, suffix=suffix,
                            reference=os.path.join(root, "Reference Videos", f"{scene}_reference.mp4")))
    return pd.DataFrame(stimuli).merge(table, on=["scene", "method"], how="left"), None


def nvsqa_stimuli(root):
    ws = openpyxl.load_workbook(os.path.join(root, "DMOS.xlsx"), data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    hdr = rows[1]
    c0 = 5
    subj_cols = [i for i in range(c0 + 2, len(hdr)) if isinstance(hdr[i], str) and hdr[i].startswith("S")]
    dmos_col = max(i for i in range(len(hdr)) if hdr[i] == "DMOS")
    recs, subj = [], []
    for r in rows[2:]:
        if r[c0] is None:
            continue
        stem = str(r[c0]).replace(".mp4", "").replace("mipnerf360", "multinerf")
        scene, suffix = stem.split("_", 1)
        recs.append(dict(scene=scene, suffix=suffix, method=NVSQA_METHODS[suffix], dmos=float(r[dmos_col])))
        subj.append(dict(scene=scene, suffix=suffix, **{hdr[i]: r[i] for i in subj_cols}))
    table = pd.DataFrame(recs)
    pub = read_sheet(os.path.join(root, "metrics.xlsx"))
    pub = pub.rename(columns={c: "pub_" + c for c in pub.columns if c not in ("scene", "method")})
    table = table.merge(pub, on=["scene", "method"], how="left")
    files = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(root, "Synthesized Videos", "**", "*.mp4"),
                                                               recursive=True)}
    refs = {os.path.basename(p)[:-len("_reference.mp4")]: p
            for p in glob.glob(os.path.join(root, "Reference Videos", "**", "*.mp4"), recursive=True)}
    table["test"] = [files[f"{s}_{x}"] for s, x in zip(table.scene, table.suffix)]
    table["reference"] = [refs[s] for s in table.scene]
    return table, pd.DataFrame(subj)


def align_lengths(test, ref):
    if len(test) == len(ref):
        return test, ref, 0
    n = min(len(test), len(ref))
    idx = np.linspace(0, n - 1, 12).astype(int)
    best = None
    for off in range(abs(len(test) - len(ref)) + 1):
        a = test[off:off + n] if len(test) > len(ref) else test
        b = ref if len(test) > len(ref) else ref[off:off + n]
        err = np.mean((a[idx, ::4, ::4].astype(np.float32) - b[idx, ::4, ::4].astype(np.float32)) ** 2)
        if best is None or err < best[0]:
            best = (err, off, a, b)
    signed = best[1] + 1 if len(test) > len(ref) else -(best[1] + 1)
    return best[2], best[3], signed


def check_labels(df):
    if "psnr" not in df or "pub_PSNR-Y" not in df or df["pub_PSNR-Y"].isna().all():
        return
    d = (df.psnr - df["pub_PSNR-Y"].astype(float))
    r = np.corrcoef(df.psnr, df["pub_PSNR-Y"].astype(float))[0, 1]
    print(f"label check: corr(our PSNR, published PSNR-Y) = {r:.4f}; mean diff {d.mean():+.2f} dB, "
          f"max |diff - mean| {np.abs(d - d.mean()).max():.2f} dB")
    if "stlpips_vgg" in df and "pub_ST-LPIPS (sheet order)" in df:
        for col in ("pub_ST-LPIPS (sheet order)", "pub_ST-LPIPS"):
            v = df[col].astype(float)
            print(f"ST-LPIPS check: {col}: corr with our ST-LPIPS (VGG) = {np.corrcoef(df.stlpips_vgg, v)[0, 1]:+.4f}, "
                  f"mean |diff| = {np.abs(df.stlpips_vgg - v).mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GS-QA", choices=["GS-QA", "NVS-QA"])
    ap.add_argument("--display", default="fhd_3h")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flip-procs", type=int, default=10)
    ap.add_argument("--extra", nargs="*", default=[], help="additional metrics, e.g. stlpips_alex stlpips_vgg")
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="compute only these metrics (e.g. flip cvvdp_image cvvdp for another viewing distance)")
    ap.add_argument("--no-descriptors", action="store_true", help="skip the distortion descriptors")
    args = ap.parse_args()
    wanted = (list(args.metrics) if args.metrics else IMAGE_METRICS + VIDEO_METRICS) + list(args.extra)
    root = os.path.join(DATA_ROOT, args.dataset)
    out_dir = P.subjective_results(args.dataset.lower().replace("-", "") +
                                   ("" if args.display == "fhd_3h" else "_" + args.display))
    for sub in ("frames", "scores"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    stimuli, subjects = gsqa_stimuli(root) if args.dataset == "GS-QA" else nvsqa_stimuli(root)
    stimuli["scene_class"] = stimuli.scene.map(SCENE_CLASS)
    if subjects is not None:
        subjects.to_csv(os.path.join(out_dir, "subjects.csv"), index=False)
    if args.limit:
        stimuli = stimuli.iloc[:args.limit]
    suite = None
    rows, ref_cache = [], {}
    t0 = time.time()
    for k, s in stimuli.reset_index(drop=True).iterrows():
        stem = f"{s.scene}_{s.suffix}"
        cache = os.path.join(out_dir, "scores", stem + ".json")
        scores = json.load(open(cache)) if os.path.exists(cache) else {}
        missing = [m for m in wanted if m not in scores]
        need_descriptors = "d_sharp_ratio" not in scores and not args.no_descriptors
        if missing or need_descriptors:
            if s.reference not in ref_cache:
                ref_cache.clear()
                ref_cache[s.reference] = video_io.read_video(s.reference)
            ref, info = ref_cache[s.reference]
            test, tinfo = video_io.read_video(s.test)
            ref_full = ref
            test, ref, offset = align_lengths(test, ref)
            if offset:
                scores["frame_offset"] = offset
                print(f"   note: {stem} has {tinfo['nb_frames']} frames, reference {len(ref_full)}; offset {offset}")
            if abs(tinfo["fps"] - info["fps"]) > 1e-6:
                scores["container_fps"] = tinfo["fps"]
                print(f"   note: {stem} is stored at {tinfo['fps']:g} fps, reference {info['fps']:g} fps")
            if missing:
                if suite is None:
                    suite = MetricSuite(display=args.display, flip_procs=args.flip_procs)
                    print(f"{len(stimuli)} stimuli, display {args.display} ({suite.ppd:.2f} ppd)", flush=True)
                new_scores, frames = suite.video(test, ref, info["fps"], names=missing)
                scores.update(new_scores, fps=info["fps"], num_frames=int(len(test)))
                npz = os.path.join(out_dir, "frames", stem + ".npz")
                if os.path.exists(npz):
                    frames = dict(np.load(npz), **frames)
                np.savez_compressed(npz, **frames)
            if need_descriptors:
                scores.update({"d_" + k2: v for k2, v in descriptors.describe(test, ref).items()})
            with open(cache, "w") as f:
                json.dump(scores, f)
            print(f"[{k + 1}/{len(stimuli)}] {s.scene:12s} {s.method:14s} " +
                  " ".join(f"{m}={v:.4g}" for m, v in scores.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
        rows.append(dict(s.drop(["test", "reference"]).to_dict(), **scores))
    if suite is not None:
        suite.close()
    df = pd.DataFrame(rows)
    check_labels(df)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print("wrote", os.path.join(out_dir, "metrics.csv"), df.shape)
if __name__ == "__main__":
    main()
