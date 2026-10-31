import concurrent.futures as cf
import json
import os
import time
import cv2
import numpy as np
import torch
import paths as _P
from . import datasets, video_io
ITER_LADDER = [500, 1000, 2000, 3000, 5000, 7000, 15000, 30000]
VIEW_DIVISORS = [2, 4, 8, 16]
GAUSS_CAPS = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
PART1_ROOT = _P.PART1_RENDERS


def all_jobs(methods=("3dgs", "nerf")):
    jobs = []
    if "3dgs" in methods:
        jobs += ["3dgs-full"] + [f"3dgs-views-{d}" for d in VIEW_DIVISORS] + [f"3dgs-gauss-{c}" for c in GAUSS_CAPS]
    if "nerf" in methods:
        jobs += ["nerf-full"] + [f"nerf-views-{d}" for d in VIEW_DIVISORS]
    return jobs


def parse_job(job):
    base, _, seed = job.partition("@")
    seed = int(seed) if seed else 0
    parts = base.split("-")
    method, kind = parts[0], parts[1]
    value = int(parts[2]) if len(parts) > 2 else None
    sfx = f"-s{seed}" if seed else ""
    if kind == "full":
        conds = [(f"iters-{n:05d}{sfx}", n) for n in ITER_LADDER]
    elif kind == "views":
        conds = [(f"views-1of{value}{sfx}", ITER_LADDER[-1])]
    elif kind == "gauss":
        conds = [(f"gauss-{value:07d}{sfx}", ITER_LADDER[-1])]
    return dict(method=method, kind=kind, value=value, seed=seed, conditions=conds)


def condition_dir(scene_spec, method, condition):
    return os.path.join(PART1_ROOT, scene_spec, method, condition)


def job_done(scene_spec, job):
    j = parse_job(job)
    return all(os.path.exists(os.path.join(condition_dir(scene_spec, j["method"], c), "meta.json"))
               for c, _ in j["conditions"])


def reference_frames(scene):
    imgs = scene.images[scene.test_ids]
    if scene.crop:
        x0, y0, w, h = scene.crop
        imgs = imgs[:, y0:y0 + h, x0:x0 + w]
    return imgs


def write_reference(scene, scene_spec):
    if scene.fps <= 0:
        return None
    path = os.path.join(PART1_ROOT, scene_spec, "reference.mp4")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        video_io.write_video(path, reference_frames(scene).numpy(), scene.fps)
    return path
@torch.no_grad()


def render_test_views(scene, model, device="cuda"):
    c2w = torch.from_numpy(scene.c2w[scene.test_ids]).float().to(device)
    K = torch.from_numpy(scene.K[scene.test_ids]).float().to(device)
    frames = []
    for i in range(len(scene.test_ids)):
        img = model.render(c2w[i], K[i], scene.width, scene.height)
        if scene.crop:
            x0, y0, w, h = scene.crop
            img = img[y0:y0 + h, x0:x0 + w]
        frames.append(img.cpu())
    return torch.stack(frames).numpy()


def save_condition(scene, out_dir, frames, meta):
    os.makedirs(out_dir, exist_ok=True)
    if scene.fps > 0:
        video_io.write_video(os.path.join(out_dir, "video.mp4"), frames, scene.fps)
    else:
        rdir = os.path.join(out_dir, "renders")
        os.makedirs(rdir, exist_ok=True)
        def write(i):
            cv2.imwrite(os.path.join(rdir, f"{i:03d}.png"), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 4])
        with cf.ThreadPoolExecutor(8) as ex:
            list(ex.map(write, range(len(frames))))
    meta = dict(meta, num_frames=int(len(frames)), height=int(frames.shape[1]), width=int(frames.shape[2]),
                fps=float(scene.fps), test_names=[scene.names[i] for i in scene.test_ids])
    with open(os.path.join(out_dir, "meta.json.tmp"), "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(os.path.join(out_dir, "meta.json.tmp"), os.path.join(out_dir, "meta.json"))


def run_job(scene, scene_spec, job, device="cuda", log=print):
    from . import gs, nerf
    j = parse_job(job)
    train_ids = scene.train_ids
    if j["kind"] == "views":
        train_ids = datasets.subsample_views(scene.train_ids, int(round(len(scene.train_ids) / j["value"])))
    steps = {s: c for c, s in j["conditions"]}
    seed = 42 + j["seed"]
    write_reference(scene, scene_spec)
    t_start = time.time()
    def snapshot(step, model, info):
        cond = steps[step]
        out_dir = condition_dir(scene_spec, j["method"], cond)
        tr = time.time()
        frames = render_test_views(scene, model, device)
        render_time = time.time() - tr
        psnr = float(np.mean([-10 * np.log10(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2) / 255 ** 2 + 1e-12)
                              for a, b in zip(frames, reference_frames(scene).numpy())]))
        meta = dict(scene=scene_spec, method=j["method"], condition=cond, job=job, kind=j["kind"],
                    value=j["value"], seed=seed, step=int(step), num_train_views=int(len(train_ids)),
                    num_train_views_full=int(len(scene.train_ids)), train_time=float(info["train_time"]),
                    render_time=float(render_time), quick_psnr=psnr)
        if "num_gaussians" in info:
            meta["num_gaussians"] = int(info["num_gaussians"])
        save_condition(scene, out_dir, frames, meta)
        log(f"  [{scene_spec} {job}] {cond}: PSNR {psnr:.2f}  train {info['train_time']:.0f}s  "
            f"render {render_time:.0f}s" + (f"  N={info['num_gaussians']}" if "num_gaussians" in info else ""))
    if j["method"] == "3dgs":
        if j["kind"] == "gauss":
            cfg = gs.GSConfig.mcmc(j["value"], seed=seed)
        else:
            cfg = gs.GSConfig(seed=seed)
        gs.train_gs(scene, train_ids, cfg, snapshot_steps=steps.keys(), snapshot_fn=snapshot, device=device,
                    log=log, log_every=0)
    elif j["method"] == "nerf":
        cfg = nerf.NerfConfig(seed=seed)
        nerf.train_nerf(scene, train_ids, cfg, snapshot_steps=steps.keys(), snapshot_fn=snapshot, device=device,
                        log=log, log_every=0)
    torch.cuda.empty_cache()
    return time.time() - t_start
