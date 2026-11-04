#!/usr/bin/env python
import argparse
import os
import sys
import time
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import torch              
from nvsqa import datasets, pipeline              


def acquire(lock):
    try:
        os.makedirs(lock)
        return True
    except FileExistsError:
        return False


def release(lock):
    os.rmdir(lock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["3dgs", "nerf"])
    ap.add_argument("--jobs", nargs="*", default=None, help="explicit job names (default: all jobs of --methods)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reverse", action="store_true", help="walk through scenes and jobs in reverse order")
    args = ap.parse_args()
    torch.set_num_threads(4)
    def log(msg):
        print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)
    jobs = args.jobs or pipeline.all_jobs(args.methods)
    if args.reverse:
        jobs = jobs[::-1]
        args.scenes = args.scenes[::-1]
    lock_root = os.path.join(pipeline.PART1_ROOT, ".locks")
    os.makedirs(lock_root, exist_ok=True)
    for spec in args.scenes:
        scene = None
        for job in jobs:
            if pipeline.job_done(spec, job):
                continue
            lock = os.path.join(lock_root, f"{spec.replace('/', '_')}__{job}")
            if not acquire(lock):
                continue
            if scene is None:
                scene = datasets.load_scene(spec)
                log(f"loaded {spec}: {tuple(scene.images.shape)}, {len(scene.train_ids)} train / "
                    f"{len(scene.test_ids)} test views")
            log(f"START {spec} {job}")
            dt = pipeline.run_job(scene, spec, job, device=args.device, log=log)
            log(f"DONE  {spec} {job} in {dt / 60:.1f} min")
            release(lock)
    log("QUEUE FINISHED")
if __name__ == "__main__":
    main()
