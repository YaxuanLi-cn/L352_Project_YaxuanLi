import concurrent.futures as cf
import os
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from . import DATA_ROOT, colmap_io
MIP360_OUTDOOR = ("bicycle", "garden", "stump", "flowers", "treehill")
MIP360_INDOOR = ("room", "counter", "kitchen", "bonsai")
MIP360_SCENES = MIP360_OUTDOOR + MIP360_INDOOR
GSQA_SCENES = {
    "m60": ("T&T", 277, 24, 1),
    "playground": ("T&T", 275, 30, 1),
    "train": ("T&T", 258, 30, 1),
    "truck": ("T&T", 226, 30, 1),
    "antique": ("IST", 351, 25, 2),
    "flowers": ("IST", 377, 25, 2),
    "playground2": ("IST", 291, 25, 2),
    "statue": ("IST", 228, 25, 2),
}
GSQA_VIDEO_SIZE = (928, 522)
@dataclass


class Scene:
    name: str
    dataset: str
    names: list
    images: torch.Tensor
    K: np.ndarray
    c2w: np.ndarray
    points: np.ndarray
    points_rgb: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    fps: float = 0.0
    crop: tuple = None
    meta: dict = field(default_factory=dict)
    @property
    def width(self):
        return self.images.shape[2]
    @property
    def height(self):
        return self.images.shape[1]
    def camera_extent(self, ids=None):
        ids = self.train_ids if ids is None else ids
        centres = self.c2w[ids, :3, 3]
        return float(np.linalg.norm(centres - centres.mean(0), axis=1).max())


def _load_image(path):
    with PILImage.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _load_images(paths, workers=16):
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(_load_image, paths))


def _poses_from_colmap(cams, images):
    ims = sorted(images.values(), key=lambda im: im.name)
    names = [im.name for im in ims]
    Ks, c2ws, sizes, dist = [], [], [], []
    for im in ims:
        cam = cams[im.camera_id]
        (fx, fy, cx, cy), d = cam.intrinsics()
        Ks.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64))
        c2ws.append(np.linalg.inv(im.w2c()))
        sizes.append((cam.width, cam.height))
        dist.append((cam.model, d))
    return names, np.stack(Ks), np.stack(c2ws), sizes, dist


def subsample_views(ids, n):
    ids = np.asarray(ids)
    if n >= len(ids):
        return ids.copy()
    pick = np.round(np.linspace(0, len(ids) - 1, n)).astype(int)
    return ids[np.unique(pick)]


def load_mip360(scene, root=None, factor=None, test_every=8):
    root = root or os.path.join(DATA_ROOT, "mipnerf360")
    factor = factor or (4 if scene in MIP360_OUTDOOR else 2)
    sdir = os.path.join(root, scene)
    cams, images, (xyz, rgb, _) = colmap_io.read_model(os.path.join(sdir, "sparse", "0"))
    names, K, c2w, sizes, _ = _poses_from_colmap(cams, images)
    imgs = _load_images([os.path.join(sdir, f"images_{factor}", n) for n in names])
    h, w = imgs[0].shape[:2]
    for i, (cw, ch) in enumerate(sizes):
        K[i, 0] *= w / cw
        K[i, 1] *= h / ch
    ids = np.arange(len(names))
    test = ids[ids % test_every == 0]
    train = ids[ids % test_every != 0]
    return Scene(scene, "mip360", names, torch.from_numpy(np.stack(imgs)), K, c2w, xyz, rgb, train, test,
                 meta={"factor": factor})


def undistort_images(imgs, K_dist, dist, K_new, size_new, device="cuda", chunk=64):
    k1, k2, p1, p2 = [float(x) for x in dist[:4]]
    wn, hn = size_new
    dev = torch.device(device)
    v, u = torch.meshgrid(torch.arange(hn, device=dev, dtype=torch.float64) + 0.5,
                          torch.arange(wn, device=dev, dtype=torch.float64) + 0.5, indexing="ij")
    x = (u - K_new[0, 2]) / K_new[0, 0]
    y = (v - K_new[1, 2]) / K_new[1, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    ud = K_dist[0, 0] * xd + K_dist[0, 2]
    vd = K_dist[1, 1] * yd + K_dist[1, 2]
    h, w = imgs.shape[1:3]
    grid = torch.stack([ud / w * 2 - 1, vd / h * 2 - 1], -1).float()[None]
    out = []
    for i in range(0, len(imgs), chunk):
        src = imgs[i:i + chunk].to(dev).permute(0, 3, 1, 2).float()
        res = F.grid_sample(src, grid.expand(len(src), -1, -1, -1), mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        out.append(res.round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu())
    return torch.cat(out)


def area_downsample(imgs, factor):
    if factor == 1:
        return imgs
    out = []
    for i in range(0, len(imgs), 64):
        x = imgs[i:i + 64].permute(0, 3, 1, 2).float()
        x = F.avg_pool2d(x, factor)
        out.append(x.round().clamp(0, 255).byte().permute(0, 2, 3, 1))
    return torch.cat(out)


def load_gsqa(scene, root=None, downsample=None, device="cuda"):
    root = root or os.path.join(DATA_ROOT, "GS-QA", "Datasets")
    group, n_train, fps, ds_default = GSQA_SCENES[scene]
    downsample = downsample or ds_default
    sdir = os.path.join(root, group, scene)
    cams_u, images_u, (xyz, rgb, _) = colmap_io.read_model(os.path.join(sdir, "sparse", "0"))
    cams_d, images_d, _ = colmap_io.read_model(os.path.join(sdir, "distorted", "sparse", "0"), cache=False)
    names, K, c2w, sizes, dist_u = _poses_from_colmap(cams_u, images_u)
    _, K_d, _, _, dist_d = _poses_from_colmap(cams_d, images_d)
    imgs = torch.from_numpy(np.stack(_load_images([os.path.join(sdir, "images", n) for n in names])))
    model, d = dist_d[0]
    if model == "OPENCV":
        imgs = undistort_images(imgs, K_d[0], d, K[0], sizes[0], device=device)
    if downsample > 1:
        wn, hn = sizes[0]
        wc, hc = wn // downsample * downsample, hn // downsample * downsample
        imgs = area_downsample(imgs[:, :hc, :wc], downsample)
        K = K.copy()
        K[:, :2] /= downsample
    ids = np.arange(len(names))
    scene_obj = Scene(scene, "gsqa", names, imgs, K, c2w, xyz, rgb, ids[:n_train], ids[n_train:], fps=float(fps),
                      meta={"group": group, "downsample": downsample})
    vw, vh = GSQA_VIDEO_SIZE
    scene_obj.crop = ((scene_obj.width - vw) // 2, (scene_obj.height - vh) // 2, vw, vh)
    return scene_obj


def load_scene(spec, **kw):
    dataset, scene = spec.split("/")
    if dataset == "mip360":
        return load_mip360(scene, **kw)
    if dataset == "gsqa":
        return load_gsqa(scene, **kw)
