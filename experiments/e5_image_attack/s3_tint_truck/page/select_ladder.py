#!/usr/bin/env python
import json
import os
import sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
PAGE = P.rec(P.E5, P.S3, "page")                                                                    
sys.path.insert(0, os.path.dirname(HERE))                      
sys.path.insert(0, P.code(P.E5, P.S2))                                                    
import ladder as LD              
import export_stimuli as E8                                              
J = LD.J
lv, floor = LD.levels()
fl = floor.set_index("scene")
out = []
for scene in LD.SCENES:
    sdn = scene.replace("/", "_")
    g = lv[lv.scene == scene].sort_values("gamma").reset_index(drop=True)
    nh = g[~g.held]
    star = nh.iloc[0] if len(nh) else g.iloc[-1]
    views = sorted(int(f[:3]) for f in os.listdir(os.path.join(star.dir, "test")) if f.endswith(".png"))
    rank = []
    for v in views:
        f = f"{v:03d}.png"
        a = np.asarray(Image.open(os.path.join(star.zero_dir, "test", f)).convert("RGB"))
        b = np.asarray(Image.open(os.path.join(star.dir, "test", f)).convert("RGB"))
        x, y, w, h, mad, _ = E8.best_window(a, b)
        rank.append((mad, v, x, y, w, h))
    rank.sort(reverse=True)
    vs = []
    for k, (mad, v, x, y, w, h) in enumerate(rank[:2]):
        f = f"{v:03d}.png"
        def cut(src, name):
            dst = os.path.join(PAGE, "img", f"L_{J.SCENE_NAME[scene]}_{k}_{name}.png")
            Image.fromarray(np.asarray(Image.open(src).convert("RGB"))[y:y + h, x:x + w]).save(dst, compress_level=6)
            return f"img/{os.path.basename(dst)}"
        vs.append(dict(view=v, crop=[x, y, w, h], ref=cut(os.path.join(LD.RERUN, sdn, "reference", "test", f), "ref"),
                       control=cut(os.path.join(LD.RERUN, sdn, "s0", "default__ctrl", "test", f), "ctrl"),
                       zero=cut(os.path.join(star.zero_dir, "test", f), "g0"),
                       levels=[cut(os.path.join(r.dir, "test", f), f"lvl{i}") for i, r in enumerate(g.itertuples())]))
    held = g[g.held]
    start = int(held.index.max()) if len(held) else 0
    out.append(dict(scene=J.SCENE_NAME[scene], start=start, views=vs,
                    floor=dict(chroma=float(fl.rms_low_chroma[scene]), area6=float(fl.area_low_gt6[scene]), areac6=float(fl.area_chroma_gt6[scene])),
                    levels=[dict(gamma=float(f"{r.gamma:g}"), verdict=J.VERDICT[r.verdict], verdict_g0=J.VERDICT[r.verdict_vs_gamma0],
                                 held=[float(r.held_psnr_over_p90), float(r.held_ssim_over_p90), float(r.held_lpips_vgg_over_p90)], z=float(r.z_dreamsim),
                                 chroma=float(r.rms_low_chroma), cof=float(r.rms_low_chroma / fl.rms_low_chroma[scene]), area6=float(r.area_low_gt6),
                                 areac6=float(r.area_chroma_gt6), src="第 10 条" if r.source != "item11" else "新补的") for r in g.itertuples()]))
json.dump(out, open(os.path.join(PAGE, "ladder.json"), "w"), ensure_ascii=False)
for s in out:
    print(s["scene"], "views", [v["view"] for v in s["views"]], "levels", [l["gamma"] for l in s["levels"]], "start", s["levels"][s["start"]]["gamma"])
