#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
os.environ.setdefault("OMP_NUM_THREADS", "8")
import helpers as C              
import uncertainty as U              
import numpy as np              
import pandas as pd              
LADDER = ["iters-00500", "iters-01000", "iters-02000", "iters-03000", "iters-05000", "iters-07000", "iters-15000",
          "iters-30000"]
COMMON7 = ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg", "flip", "cvvdp_image"]
PIXEL_METRICS = ["l1", "mse", "ssim", "lpips_vgg", "lpips_alex", "flip"]


class Ctx:
    def __init__(self, args):
        self.pcfg = C.load_cfg(os.path.join(HERE, "configs", "pilot.yaml"))
        self.rcfg = C.load_cfg(P.code(P.E4, "rerun", "configs", "rerun.yaml"))
        import yaml
        self.seed_jobs = __import__("train_seeds").plan(yaml.safe_load(open(os.path.join(HERE, "configs", "pilot.yaml"))))
        self.out = args.out or P.PILOT
        self.stim = self.rcfg["stimuli_root"]
        self.t = self.pcfg["tile"]
        self.q = self.pcfg["quintiles"]
        self.scenes = args.scenes or self.pcfg["scenes"]
        self.max_views = args.max_views
        self.metric_names = args.metrics
        self.models_only = args.models
        self.allow_missing_seeds = args.allow_missing_seeds
        self.loop = getattr(args, "loop", False)
        os.makedirs(self.out, exist_ok=True)
    def views(self, spec):
        v = json.load(open(os.path.join(self.rcfg["views_dir"], f"{C.scene_dir(spec)}.json")))
        held, tr = v["heldout"], v["train"]
        if self.max_views:
            held, tr = held[:self.max_views], tr[:self.max_views]
        return {"test": [(int(k), f"{k:03d}.png") for k in held], "train": [(int(i), f"{i:04d}.png") for i in tr]}
    def seeds(self, spec):
        return [s for sp, s, _, _ in self.seed_jobs if sp == spec]
    def loss_seeds(self, spec):
        return self.rcfg["seed_scene_seeds"] if spec == self.rcfg["seed_scene"] else self.rcfg["seeds"]
    def models(self):
        m = ["default__ctrl"] + [f"{l}__pure" for l in self.rcfg["pure"]] + [f"{l}__mixed" for l in self.rcfg["mixed"]]
        return [x for x in m if not self.models_only or x in self.models_only]
    def png(self, spec, *parts):
        return os.path.join(self.stim, C.scene_dir(spec), *parts)


def load(paths):
    return C.read_pngs(paths)


def strata_file(ctx, spec):
    return os.path.join(ctx.out, "strata", f"{C.scene_dir(spec)}.parquet")


def step_strata(ctx):
    os.makedirs(os.path.join(ctx.out, "strata"), exist_ok=True)
    for spec in ctx.scenes:
        f = strata_file(ctx, spec)
        if os.path.exists(f):
            continue
        seeds = ctx.seeds(spec)
        have = [s for s in seeds if os.path.exists(ctx.png(spec, "seeds", str(s), "done"))]
        if len(have) < len(seeds):
            C.log(f"strata {spec}: renders of seeds {sorted(set(seeds) - set(have))} missing"
                  + ("; using the others (test run)" if ctx.allow_missing_seeds else "; scene skipped"))
            if not ctx.allow_missing_seeds or len(have) < 2:
                continue
            seeds = have
        if not os.path.exists(ctx.png(spec, "reference", "done")):
            C.log(f"strata {spec}: reference views not stored yet; scene skipped")
            continue
        rows, thr = [], []
        for split, views in ctx.views(spec).items():
            recs = []
            for vid, name in views:
                ref = load([ctx.png(spec, "reference", split, name)])[0]
                rend = load([ctx.png(spec, "seeds", str(s), split, name) for s in seeds])
                V = U.variance_map(rend)
                st = U.tile_stats(V, U.bias2_map(rend, ref, V), U.texture_map(ref), ctx.t)
                for t in range(len(st["U"])):
                    recs.append(dict(scene=spec, split=split, view=vid, tile=t, U=st["U"][t], B=st["B"][t],
                                     G=st["G"][t], K=len(seeds)))
            d = pd.DataFrame(recs)
            d["quintile"], edges = U.assign_strata(d.U.to_numpy(), ctx.q)
            rows.append(d)
            for k in range(ctx.q):
                thr.append(dict(scene=spec, split=split, quintile=k + 1, U_min=edges[k, 0], U_max=edges[k, 1],
                                n_tiles=int((d.quintile == k + 1).sum()), seeds=" ".join(map(str, seeds)),
                                written=time.strftime("%Y-%m-%d %H:%M:%S")))
            C.log(f"strata {spec} {split}: {len(d)} tiles, K = {len(seeds)}")
        pd.DataFrame(thr).to_csv(f[:-8] + "_thresholds.csv", index=False)
        pd.concat(rows).to_parquet(f + ".tmp", index=False)
        os.replace(f + ".tmp", f)
    files = [strata_file(ctx, s) for s in ctx.scenes if os.path.exists(strata_file(ctx, s))]
    if files:
        pd.concat([pd.read_parquet(x) for x in files]).to_parquet(os.path.join(ctx.out, "tiles_meta.parquet"), index=False)
        pd.concat([pd.read_csv(x[:-8] + "_thresholds.csv") for x in files]).to_csv(
            os.path.join(ctx.out, "strata_thresholds.csv"), index=False)
        C.log(f"thresholds of {len(files)} scenes: {os.path.join(ctx.out, 'strata_thresholds.csv')}")


class TileMetrics:
    def __init__(self, names):
        import torch
        torch.backends.cudnn.benchmark = True
        from gh import metrics as M
        self.torch, self.M = torch, M
        frac = float(os.environ.get("TILE_MEM_FRACTION", "0"))
        if frac > 0:
            torch.cuda.set_per_process_memory_fraction(frac)
        self.m = M.Metrics()
        self.names = names
    def cvvdp_batch(self, t, r):
        jod, _ = self.m.cvvdp.predict(t.permute(0, 2, 3, 1).contiguous(), r.permute(0, 2, 3, 1).contiguous(),
                                      dim_order="BHWC")
        return jod.reshape(-1)
    def __call__(self, test_tiles, ref_tiles, batch=32):
        torch = self.torch
        out = {n: [] for n in self.names}
        with torch.no_grad():
            for i in range(0, len(test_tiles), batch):
                t = torch.from_numpy(test_tiles[i:i + batch]).cuda().permute(0, 3, 1, 2).float() / 255.0
                r = torch.from_numpy(ref_tiles[i:i + batch]).cuda().permute(0, 3, 1, 2).float() / 255.0
                for n in self.names:
                    v = self.cvvdp_batch(t, r) if n == "cvvdp_image" else self.m(n, t, r)
                    out[n].append(v.double().cpu().numpy())
        res = {}
        for n, v in out.items():
            v = np.concatenate(v)
            if n == "psnr":
                v = np.minimum(v, 60.0)
            res[n] = self.M.SIGN[n] * v
        return res
    def whole(self, test, ref):
        vals = self.m.evaluate(test, ref, self.names)
        return {n: self.M.SIGN[n] * np.asarray(v) for n, v in vals.items()}


def tiles_of(imgs, t):
    return np.concatenate([U.tile_view(im, t) for im in imgs])


def step_metrics(ctx):
    import torch
    from gh import metrics as M
    names = ctx.metric_names or M.IMAGE_METRICS
    tm = TileMetrics(names)
    refs, told = {}, set()
    while True:
        pending = 0
        for spec in ctx.scenes:
            if not os.path.exists(strata_file(ctx, spec)):
                if spec not in told:
                    C.log(f"tile metrics {spec}: strata not written yet; scene skipped")
                    told.add(spec)
                pending += 1
                continue
            for split, views in ctx.views(spec).items():
                for seed in ctx.loss_seeds(spec):
                    for model in ctx.models():
                        f = os.path.join(ctx.out, "cache", C.scene_dir(spec), split, f"s{seed}", f"{model}.parquet")
                        if os.path.exists(f):
                            continue
                        rd = os.path.join(ctx.rcfg["runs_root"], C.scene_dir(spec), f"s{seed}", model)
                        if not os.path.exists(os.path.join(rd, "done.json")):
                            pending += 1
                            continue
                        os.makedirs(os.path.dirname(f), exist_ok=True)
                        if not C.acquire(f + ".lock"):
                            pending += 1
                            continue
                        if (spec, split) not in refs:
                            ref = load([ctx.png(spec, "reference", split, n) for _, n in views])
                            refs[(spec, split)] = (ref, tiles_of(ref, ctx.t))
                        ref, rt = refs[(spec, split)]
                        n_per = rt.shape[0] // len(views)
                        img = load([ctx.png(spec, f"s{seed}", model, split, n) for _, n in views])
                        vals = tm(tiles_of(img, ctx.t), rt)
                        w = tm.whole(img, ref) if split == "train" and seed == 0 else None
                        vid = np.repeat([v for v, _ in views], n_per)
                        tile = np.tile(np.arange(n_per), len(views))
                        df = pd.concat([pd.DataFrame(dict(view=vid, tile=tile, metric=n, value=v)) for n, v in vals.items()])
                        if w is not None:
                            df = pd.concat([df, pd.concat([pd.DataFrame(dict(view=[v for v, _ in views], tile=-1,
                                                                             metric=n, value=x)) for n, x in w.items()])])
                        df.to_parquet(f, index=False)
                        C.log(f"tile metrics {spec} {split} s{seed} {model}: {len(vid)} tiles")
                        C.release(f + ".lock")
                        torch.cuda.empty_cache()
        if not ctx.loop or pending == 0:
            break
        time.sleep(45)
    tm.m = None


def step_ladder(ctx):
    from nvsqa import pipeline, video_io
    tm = TileMetrics(ctx.metric_names or COMMON7)
    for spec in ctx.scenes:
        views = ctx.views(spec)["test"]
        ks = [k for k, _ in views]
        base = os.path.join(pipeline.PART1_ROOT, spec, "3dgs")
        video = spec.startswith("gsqa/")
        if video:
            ref = video_io.read_video(os.path.join(pipeline.PART1_ROOT, spec, "reference.mp4"))[0][ks]
        else:
            ref = load([ctx.png(spec, "reference", "test", n) for _, n in views])
        rt = tiles_of(ref, ctx.t)
        n_per = rt.shape[0] // len(views)
        for cond in [c for lv in LADDER for c in (lv, lv + "-s1")]:
            d = os.path.join(base, cond)
            if not os.path.isdir(d):
                continue
            f = os.path.join(ctx.out, "cache_ladder", C.scene_dir(spec), f"{cond}.parquet")
            if os.path.exists(f):
                continue
            img = video_io.read_video(os.path.join(d, "video.mp4"))[0][ks] if video else load(
                [os.path.join(d, "renders", n) for _, n in views])
            vals = tm(tiles_of(img, ctx.t), rt)
            df = pd.concat([pd.DataFrame(dict(view=np.repeat(ks, n_per), tile=np.tile(np.arange(n_per), len(views)),
                                              metric=n, value=v)) for n, v in vals.items()])
            os.makedirs(os.path.dirname(f), exist_ok=True)
            df.to_parquet(f, index=False)
            C.log(f"ladder {spec} {cond}")
BORDER = 5


def make_pixel_maps():
    import lpips
    import torch
    import torch.nn.functional as F
    from gh import flip_torch
    from gh import metrics as M
    from nvsqa import metrics as NM
    torch.backends.cudnn.benchmark = True
    nets = {n: lpips.LPIPS(net=n.split("_")[1], spatial=True, verbose=False).cuda().eval() for n in ("lpips_vgg", "lpips_alex")}
    ppd = M.Metrics().ppd
    b = BORDER
    def maps(t8, r8):
        t = torch.from_numpy(t8).cuda().permute(2, 0, 1)[None].float() / 255
        r = torch.from_numpy(r8).cuda().permute(2, 0, 1)[None].float() / 255
        out = {}
        with torch.no_grad():
            out["l1"] = -(t - r).abs().mean(1)[0]
            out["mse"] = -((t - r) ** 2).mean(1)[0]
            c = 3
            g = NM._gauss(11, 1.5, t.device)
            kh, kv = g.view(1, 1, 1, 11).expand(c, 1, 1, 11), g.view(1, 1, 11, 1).expand(c, 1, 11, 1)
            blur = lambda x: F.conv2d(F.conv2d(x, kh, groups=c), kv, groups=c)              
            mx, my = blur(t), blur(r)
            sxx, syy, sxy = blur(t * t) - mx ** 2, blur(r * r) - my ** 2, blur(t * r) - mx * my
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            sm = (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean(1)[0]
            out["ssim"] = F.pad(sm, (b, b, b, b))
            for n, net in nets.items():
                out[n] = -net(t, r, normalize=True)[0, 0]
            out["flip"] = -flip_torch.compute_ldrflip(flip_torch.color_space_transform(t.clamp(0, 1), "srgb2ycxcz"),
                                                      flip_torch.color_space_transform(r.clamp(0, 1), "srgb2ycxcz"),
                                                      ppd, 0.7, 0.5, 0.4, 0.95, 1e-15)[0, 0]
        return {k: v[b:-b, b:-b].double().cpu().numpy() for k, v in out.items()}
    return maps


def step_pixel(ctx):
    import cv2
    maps = make_pixel_maps()
    sigma = ctx.pcfg["pixel_sigma"]
    b = BORDER
    for spec in ctx.scenes:
        seeds = ctx.seeds(spec)
        for split, views in ctx.views(spec).items():
            f = os.path.join(ctx.out, "pixel", f"{C.scene_dir(spec)}_{split}.parquet")
            if os.path.exists(f):
                continue
            refs = {vid: load([ctx.png(spec, "reference", split, n)])[0] for vid, n in views}
            vs = {}
            for vid, n in views:
                V = U.variance_map(load([ctx.png(spec, "seeds", str(s), split, n) for s in seeds]))
                vs[vid] = cv2.GaussianBlur(V, (0, 0), sigma)[b:-b, b:-b]
            allv = np.concatenate([v.ravel() for v in vs.values()])
            order = np.argsort(allv, kind="stable")
            rank = np.empty(len(allv), np.int64)
            rank[order] = np.arange(len(allv))
            strata = (rank * ctx.q) // len(allv) + 1
            sizes = [v.size for v in vs.values()]
            split_strata = dict(zip(vs, np.split(strata, np.cumsum(sizes)[:-1])))
            rows = []
            cnt = np.bincount(strata - 1, minlength=ctx.q)
            for seed in ctx.loss_seeds(spec):
                for model in ctx.models():
                    paths = [ctx.png(spec, f"s{seed}", model, split, n) for _, n in views]
                    acc = {m: np.zeros(ctx.q) for m in PIXEL_METRICS}
                    for (vid, _), p in zip(views, paths):
                        mp = maps(load([p])[0], refs[vid])
                        st = split_strata[vid] - 1
                        for m in PIXEL_METRICS:
                            acc[m] += np.bincount(st, weights=mp[m].ravel(), minlength=ctx.q)
                    for m in PIXEL_METRICS:
                        for k in range(ctx.q):
                            rows.append(dict(scene=spec, split=split, seed=seed, model=model, metric=m,
                                             quintile=k + 1, value=acc[m][k] / cnt[k], n_pixels=int(cnt[k])))
            os.makedirs(os.path.dirname(f), exist_ok=True)
            pd.DataFrame(rows).to_parquet(f, index=False)
            C.log(f"pixel strata {spec} {split}")


def step_checks(ctx):
    import torch
    from gh import metrics as M
    spec = ctx.scenes[0]
    seed = ctx.seeds(spec)[0]
    views = ctx.views(spec)["test"]
    idx = [views[i] for i in np.linspace(0, len(views) - 1, 8).round().astype(int)]
    ref = load([ctx.png(spec, "reference", "test", n) for _, n in idx])
    img = load([ctx.png(spec, "seeds", str(seed), "test", n) for _, n in idx])
    rt, tt = tiles_of(ref, ctx.t)[:64], tiles_of(img, ctx.t)[:64]
    tm = TileMetrics(["cvvdp_image"])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    with torch.no_grad():
        t = torch.from_numpy(tt).cuda().permute(0, 3, 1, 2).float() / 255
        r = torch.from_numpy(rt).cuda().permute(0, 3, 1, 2).float() / 255
        a = np.concatenate([tm.cvvdp_batch(t[i:i + 32], r[i:i + 32]).double().cpu().numpy() for i in range(0, len(t), 32)])
        b = tm.m("cvvdp_image", t, r).double().cpu().numpy()
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
    os.makedirs(os.path.join(ctx.out, "checks"), exist_ok=True)
    json.dump(dict(scene=spec, seed=seed, n_tiles=len(a), max_abs_diff_jod=float(np.abs(a - b).max()),
                   mean_jod=float(b.mean()), cudnn="deterministic"),
              open(os.path.join(ctx.out, "checks", "cvvdp_batch.json"), "w"), indent=1)
    names = list(M.IMAGE_METRICS)
    rows = []
    cb = torch.backends.cudnn
    for sp in ctx.scenes:
        vs = ctx.views(sp)["test"]
        pick = [vs[0], vs[len(vs) // 2]]
        sd = ctx.seeds(sp)[0]
        rr = tiles_of(load([ctx.png(sp, "reference", "test", n) for _, n in pick]), ctx.t)
        ii = tiles_of(load([ctx.png(sp, "seeds", str(sd), "test", n) for _, n in pick]), ctx.t)
        res = {}
        for mode in ("default", "benchmark"):
            cb.benchmark, cb.deterministic = (mode == "benchmark"), False
            out = {n: [] for n in names}
            with torch.no_grad():
                for i in range(0, len(ii), 32):
                    t = torch.from_numpy(ii[i:i + 32]).cuda().permute(0, 3, 1, 2).float() / 255
                    r = torch.from_numpy(rr[i:i + 32]).cuda().permute(0, 3, 1, 2).float() / 255
                    for n in names:
                        v = tm.cvvdp_batch(t, r) if (n == "cvvdp_image" and mode == "benchmark") else tm.m(n, t, r)
                        out[n].append(v.double().cpu().numpy())
            res[mode] = {n: np.concatenate(v) for n, v in out.items()}
        for n in names:
            a_, b_ = res["default"][n], res["benchmark"][n]
            rows.append(dict(scene=sp, metric=n, n_tiles=len(a_), mean_default=float(a_.mean()),
                             max_abs_diff=float(np.abs(a_ - b_).max()), abs_diff_of_means=float(abs(a_.mean() - b_.mean())),
                             sd_over_tiles=float(a_.std())))
    cb.benchmark, cb.deterministic = True, False
    pd.DataFrame(rows).to_csv(os.path.join(ctx.out, "checks", "cudnn_benchmark.csv"), index=False)
    maps = make_pixel_maps()
    m = M.Metrics()
    bd = BORDER
    rows = []
    for (vid, _), t8, r8 in list(zip(idx, img, ref))[:3]:
        mp = maps(t8, r8)
        full = m.evaluate(t8[None], r8[None], ["ssim"])
        crop = m.evaluate(t8[None, bd:-bd, bd:-bd], r8[None, bd:-bd, bd:-bd], ["l1", "mse", "lpips_vgg", "lpips_alex", "flip"])
        for n in PIXEL_METRICS:
            v = M.SIGN[n] * mp[n].mean()
            ref_v = full["ssim"][0] if n == "ssim" else crop[n][0]
            rows.append(dict(scene=spec, view=vid, metric=n, map_mean=v, metric_value=float(ref_v),
                             compared_with="whole image" if n == "ssim" else f"image cropped by {bd} px",
                             rel_diff=abs(v - ref_v) / abs(ref_v)))
    pd.DataFrame(rows).to_csv(os.path.join(ctx.out, "checks", "pixel_maps.csv"), index=False)
    C.log(f"checks written to {os.path.join(ctx.out, 'checks')}")


def step_table(ctx):
    meta = pd.read_parquet(os.path.join(ctx.out, "tiles_meta.parquet"))
    parts = []
    for f in sorted(glob.glob(os.path.join(ctx.out, "cache", "*", "*", "s*", "*.parquet"))):
        sc, split, seed, model = f.split(os.sep)[-4:]
        d = pd.read_parquet(f)
        d = d[d.tile >= 0]
        d["scene"], d["split"], d["seed"], d["model"] = sc.replace("_", "/", 1), split, int(seed[1:]), model[:-8]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df.merge(meta[["scene", "split", "view", "tile", "U", "B", "G", "quintile"]],
                  on=["scene", "split", "view", "tile"], how="left")
    cols = ["scene", "split", "view", "tile", "model", "seed", "metric", "value", "U", "B", "G", "quintile"]
    df[cols].to_parquet(os.path.join(ctx.out, "tiles_long.parquet"), index=False)
    C.log(f"long table: {len(df)} rows -> {os.path.join(ctx.out, 'tiles_long.parquet')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["strata", "metrics", "ladder", "pixel", "table", "checks"])
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--metrics", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--loop", action="store_true", help="metrics: wait for runs that are still training")
    ap.add_argument("--allow-missing-seeds", action="store_true", help="test runs only: strata from the seeds that exist")
    args = ap.parse_args()
    ctx = Ctx(args)
    dict(strata=step_strata, metrics=step_metrics, ladder=step_ladder, pixel=step_pixel, table=step_table,
         checks=step_checks)[args.step](ctx)
if __name__ == "__main__":
    main()
