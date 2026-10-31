#!/usr/bin/env python
import argparse
import io
import os
import time
import urllib.request
import zipfile
URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
OUTDOOR = {"bicycle", "garden", "stump", "flowers", "treehill"}
INDOOR = {"room", "counter", "kitchen", "bonsai"}


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block=8 << 20, max_blocks=24):
        self.url, self.block, self.max_blocks = url, block, max_blocks
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache = {}
        self.nbytes = 0
    def seekable(self):
        return True
    def readable(self):
        return True
    def tell(self):
        return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos
    def _get_block(self, i):
        if i in self.cache:
            return self.cache[i]
        lo = i * self.block
        hi = min(self.size, lo + self.block) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.nbytes += len(data)
        if len(self.cache) >= self.max_blocks:
            self.cache.pop(next(iter(self.cache)))
        self.cache[i] = data
        return data
    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.block)
            chunk = self._get_block(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)
    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["garden", "bicycle", "kitchen", "bonsai"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--list", action="store_true", help="only list the archive content summary")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(args.root or os.path.join(here, "..", "..", "dataset", "mipnerf360"))
    f = HTTPRangeFile(URL)
    zf = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    infos = zf.infolist()
    if args.list:
        import collections
        agg = collections.Counter()
        cnt = collections.Counter()
        for zi in infos:
            k = "/".join(zi.filename.split("/")[:2])
            agg[k] += zi.file_size
            cnt[k] += 1
        for k in sorted(agg):
            print(f"{agg[k] / 1e6:10.1f} MB {cnt[k]:5d}  {k}")
        return
    for scene in args.scenes:
        factor = 4 if scene in OUTDOOR else 2
        keep = (f"{scene}/images_{factor}/", f"{scene}/sparse/", f"{scene}/poses_bounds.npy")
        todo = [zi for zi in infos if zi.filename.startswith(keep) and not zi.is_dir()]
        todo.sort(key=lambda zi: zi.header_offset)
        total = sum(zi.file_size for zi in todo)
        print(f"[{scene}] {len(todo)} files, {total / 1e6:.0f} MB (images_{factor})", flush=True)
        t0 = time.time()
        for k, zi in enumerate(todo):
            dst = os.path.join(root, zi.filename)
            if os.path.exists(dst) and os.path.getsize(dst) == zi.file_size:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(zi) as src, open(dst, "wb") as out:
                out.write(src.read())
            if (k + 1) % 100 == 0:
                print(f"   {k + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        print(f"[{scene}] done in {time.time() - t0:.0f}s", flush=True)
    print(f"transferred {f.nbytes / 1e9:.2f} GB")
if __name__ == "__main__":
    main()
