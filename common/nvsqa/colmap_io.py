import os
import struct
from dataclasses import dataclass
import numpy as np
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
@dataclass


class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray
    def intrinsics(self):
        p = self.params
        if self.model == "SIMPLE_PINHOLE":
            return (p[0], p[0], p[1], p[2]), p[3:]
        if self.model == "PINHOLE":
            return (p[0], p[1], p[2], p[3]), p[4:]
        if self.model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return (p[0], p[0], p[1], p[2]), p[3:]
        if self.model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
            return (p[0], p[1], p[2], p[3]), p[4:]
@dataclass


class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    def w2c(self):
        m = np.eye(4)
        m[:3, :3] = qvec2rotmat(self.qvec)
        m[:3, 3] = self.tvec
        return m


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            name, npar = CAMERA_MODELS[model_id]
            params = np.array(struct.unpack("<" + "d" * npar, f.read(8 * npar)))
            cams[cam_id] = Camera(cam_id, name, w, h, params)
    return cams


def read_images(path):
    images = {}
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    (n,) = struct.unpack_from("<Q", data, off)
    off += 8
    for _ in range(n):
        image_id = struct.unpack_from("<i", data, off)[0]
        qvec = np.array(struct.unpack_from("<4d", data, off + 4))
        tvec = np.array(struct.unpack_from("<3d", data, off + 36))
        camera_id = struct.unpack_from("<i", data, off + 60)[0]
        off += 64
        end = data.index(b"\x00", off)
        name = data[off:end].decode("utf-8")
        off = end + 1
        (n2d,) = struct.unpack_from("<Q", data, off)
        off += 8 + 24 * n2d
        images[image_id] = Image(image_id, qvec, tvec, camera_id, name)
    return images


def read_points3d(path):
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    (n,) = struct.unpack_from("<Q", data, off)
    off += 8
    xyz = np.empty((n, 3), np.float64)
    rgb = np.empty((n, 3), np.uint8)
    err = np.empty((n,), np.float64)
    for i in range(n):
        xyz[i] = struct.unpack_from("<3d", data, off + 8)
        rgb[i] = struct.unpack_from("<3B", data, off + 32)
        err[i] = struct.unpack_from("<d", data, off + 35)[0]
        (track_len,) = struct.unpack_from("<Q", data, off + 43)
        off += 51 + 8 * track_len
    return xyz.astype(np.float32), rgb, err.astype(np.float32)


def read_model(sparse_dir, cache=True):
    cams = read_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images(os.path.join(sparse_dir, "images.bin"))
    cache_file = os.path.join(sparse_dir, "points3D_cache.npz")
    if cache and os.path.exists(cache_file):
        z = np.load(cache_file)
        xyz, rgb, err = z["xyz"], z["rgb"], z["err"]
    else:
        xyz, rgb, err = read_points3d(os.path.join(sparse_dir, "points3D.bin"))
        if cache:
            np.savez(cache_file, xyz=xyz, rgb=rgb, err=err)
    return cams, images, (xyz, rgb, err)
