import json
import os
import time
import numpy as np
import yaml
import paths as P

ROOT = P.ROOT


def log(msg):
    print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def load_cfg(path):
    cfg = yaml.safe_load(open(path))
    for k, v in list(cfg.items()):
        if isinstance(v, str) and (k.endswith("_root") or k.endswith("_dir") or k.endswith("_ckpts")):
            cfg[k] = os.path.join(ROOT, v)
    return cfg


def scene_dir(spec):
    return spec.replace("/", "_")


def select_views(scene, max_heldout=60, n_train=30, seed=0):
    n = len(scene.test_ids)
    held = np.arange(n) if n <= max_heldout else np.unique(np.round(np.linspace(0, n - 1, max_heldout)).astype(int))
    rng = np.random.default_rng(seed)
    tr = np.sort(rng.choice(np.asarray(scene.train_ids), size=n_train, replace=False))
    return dict(scene=f"{scene.dataset}/{scene.name}", n_test=int(n), heldout=[int(x) for x in held],
                test_ids=[int(x) for x in scene.test_ids], train=[int(x) for x in tr],
                test_names=[scene.names[i] for i in scene.test_ids], train_names=[scene.names[i] for i in tr])


def views_for(scene, spec, cfg):
    os.makedirs(cfg["views_dir"], exist_ok=True)
    f = os.path.join(cfg["views_dir"], f"{scene_dir(spec)}.json")
    if os.path.exists(f):
        return json.load(open(f))
    v = select_views(scene, cfg["max_heldout_views"], cfg["train_views"], cfg["train_view_seed"])
    tmp = f + f".{os.getpid()}"
    json.dump(v, open(tmp, "w"), indent=1)
    os.replace(tmp, f)
    return v


def write_pngs(d, frames, names, level=3):
    import concurrent.futures as cf
    import cv2
    os.makedirs(d, exist_ok=True)

    def w(i):
        cv2.imwrite(os.path.join(d, names[i]), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, level])
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(w, range(len(frames))))


def read_pngs(paths):
    import concurrent.futures as cf
    import cv2

    def r(p):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    with cf.ThreadPoolExecutor(8) as ex:
        return np.stack(list(ex.map(r, paths)))


def acquire(lock):
    try:
        os.makedirs(lock)
        return True
    except FileExistsError:
        return False


def release(lock):
    os.rmdir(lock)
