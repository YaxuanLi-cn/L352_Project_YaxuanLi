import concurrent.futures as cf
import os
import shutil
import cv2
import numpy as np
import torch
from nvsqa import pipeline, video_io
from . import metrics as M
@torch.no_grad()


def render_views(model, scene, ids, device="cuda"):
    c2w = torch.from_numpy(scene.c2w[ids]).float().to(device)
    K = torch.from_numpy(scene.K[ids]).float().to(device)
    out = []
    for i in range(len(ids)):
        img = model.render(c2w[i], K[i], scene.width, scene.height)
        if scene.crop:
            x0, y0, w, h = scene.crop
            img = img[y0:y0 + h, x0:x0 + w]
        out.append(img.cpu())
    return torch.stack(out).numpy()


def crop_ref(scene, ids):
    imgs = scene.images[ids]
    if scene.crop:
        x0, y0, w, h = scene.crop
        imgs = imgs[:, y0:y0 + h, x0:x0 + w]
    return imgs.numpy()


def write_pngs(d, frames, names=None):
    os.makedirs(d, exist_ok=True)
    def w(i):
        fn = names[i] if names else f"{i:03d}.png"
        cv2.imwrite(os.path.join(d, fn), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(w, range(len(frames))))


def evaluate(model, scene, spec, key, metrics, train_subset, image_metrics, run_dir, stim_dir=None):
    rows = []
    def add(split, vid, metric, value):
        rows.append(dict(key, split=split, view_id=int(vid), metric=metric, value=float(value)))
    for split, ids in (("test", scene.test_ids), ("train", train_subset)):
        test = render_views(model, scene, ids)
        ref = crop_ref(scene, ids)
        vals = metrics.evaluate(test, ref, image_metrics)
        for m, v in vals.items():
            for j, x in enumerate(v):
                add(split, j, m, x)
        bias = (test.astype(np.float32) - ref.astype(np.float32)).mean(axis=(1, 2))
        for j in range(len(ids)):
            for c, ch in enumerate("rgb"):
                add(split, j, f"diag_bias_{ch}", bias[j, c])
        if split == "test":
            if scene.fps > 0:
                rows += _path(test, scene, spec, metrics, run_dir, stim_dir, add)
            elif stim_dir:
                write_pngs(stim_dir, test)
        del test
    add("model", -1, "num_gaussians", model.num_gaussians)
    torch.cuda.empty_cache()
    return rows


def _path(frames, scene, spec, metrics, run_dir, stim_dir, add):
    tmp = os.path.join(run_dir, "video.tmp.mp4")
    video_io.write_video(tmp, frames, scene.fps)
    dec = video_io.read_video(tmp)[0]
    ref = video_io.read_video(os.path.join(pipeline.PART1_ROOT, spec, "reference.mp4"))[0]
    v = metrics.video(dec, ref, scene.fps)
    add("path", -1, "cvvdp", v["cvvdp"])
    add("path", -1, "cvvdp_sust", v["cvvdp_sust"])
    per = metrics.evaluate(dec, ref, ["cvvdp_image"])["cvvdp_image"]
    for j, x in enumerate(per):
        add("path", j, "cvvdp_image", x)
    if stim_dir:
        os.makedirs(stim_dir, exist_ok=True)
        shutil.move(tmp, os.path.join(stim_dir, "video.mp4"))
    else:
        os.remove(tmp)
    return []
