import json
import subprocess
import numpy as np


def probe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
           "stream=width,height,r_frame_rate,nb_read_frames,codec_name,pix_fmt,bit_rate,profile",
           "-of", "json", path]
    s = json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": int(s["width"]), "height": int(s["height"]), "fps": float(num) / float(den),
        "nb_frames": int(s["nb_read_frames"]), "codec": s["codec_name"], "pix_fmt": s["pix_fmt"],
        "profile": s.get("profile"), "bit_rate": int(s.get("bit_rate", 0) or 0),
    }


def read_video(path, max_frames=None):
    info = probe(path)
    w, h = info["width"], info["height"]
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return frames, info


def write_video(path, frames, fps, lossless=True, crf=12):
    frames = np.ascontiguousarray(frames)
    f, h, w, _ = frames.shape
    quality = ["-qp", "0"] if lossless else ["-crf", str(crf)]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", *quality,
           "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    return path


def yuv420_roundtrip(frames, fps=30):
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tmp.mp4")
        write_video(p, frames, fps)
        return read_video(p)[0]
