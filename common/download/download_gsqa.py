#!/usr/bin/env python
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.parse
import urllib.request
API = "https://api.github.com/repos/pedrogcmartin/{repo}/git/trees/HEAD?recursive=1"
RAW = "https://raw.githubusercontent.com/pedrogcmartin/{repo}/main/{path}"


def fetch_tree(repo, cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    with urllib.request.urlopen(API.format(repo=repo), timeout=120) as r:
        tree = json.load(r)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(tree, open(cache, "w"))
    return tree


def download(repo, path, size, root):
    dst = os.path.join(root, path)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = RAW.format(repo=repo, path=urllib.parse.quote(path))
    with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="GS-QA", choices=["GS-QA", "NVS-QA"])
    ap.add_argument("--root", default=None, help="destination (default: <project>/dataset/<repo>)")
    ap.add_argument("--videos", action="store_true", help="synthesized + reference videos and score sheets")
    ap.add_argument("--scenes", nargs="*", default=[], help="GS-QA training data of these scenes")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = args.root or os.path.join(here, "..", "..", "dataset", args.repo)
    root = os.path.abspath(root)
    tree = fetch_tree(args.repo, os.path.join(root, ".tree.json"))
    wanted = []
    for e in tree["tree"]:
        if e["type"] != "blob":
            continue
        p = e["path"]
        low = p.lower()
        if low.endswith(".ds_store"):
            continue
        top = p.split("/")[0]
        take = False
        if args.videos:
            take |= top in ("Synthesized Videos", "Reference Videos", "docs")
            take |= "/" not in p
        for s in args.scenes:
            take |= p.startswith(f"Datasets/T&T/{s}/") or p.startswith(f"Datasets/IST/{s}/")
        if take:
            wanted.append((p, e["size"]))
    total = sum(s for _, s in wanted)
    print(f"{len(wanted)} files, {total / 1e9:.2f} GB -> {root}", flush=True)
    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download, args.repo, p, s, root): (p, s) for p, s in wanted}
        for i, fut in enumerate(cf.as_completed(futs)):
            p, s = futs[fut]
            fut.result()
            done += s
            if (i + 1) % 50 == 0 or i + 1 == len(wanted):
                print(f"  {i + 1}/{len(wanted)} files, {done / 1e9:.2f}/{total / 1e9:.2f} GB, "
                      f"{time.time() - t0:.0f}s", flush=True)
    print("done")
if __name__ == "__main__":
    main()
