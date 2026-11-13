#!/usr/bin/env python
import argparse
import glob
import json
import os
import pandas as pd
import build as B
P = B.P
VIEWS = [4, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=B.OUT)
    args = ap.parse_args()
    B.OUT = args.out
    mf = os.path.join(args.out, "manifest.json")
    man = json.load(open(mf))
    have = set(t["scene"] for t in man["trials"] if t["kind"] == "ext")
    df = pd.read_csv(os.path.join(P.S6_REC, "results", "ext_runs.csv"))
    cand = df[(df.target == "multi") & df.held.astype(bool) & (df.hold_extra.fillna("") == "")]
    chosen = dict(man.get("ext", {}).get("chosen", {}))
    for s in B.SCENES:
        sdn = s.replace("/", "_")
        if s in have:
            print(s, "keeps", chosen.get(s, {}).get("run"))
            continue
        g = cand[cand.scene == s]
        if not len(g):
            print(s, "no multi run holds the extended set -> no ext trial")
            continue
        r = g.loc[g.tint.idxmax()]
        d = [os.path.join(P.S6_RUNS, sdn, r.run)]
        name = r.run
        hv = B.held_views(sdn)
        chosen[s] = dict(run=name, gamma=float(r.gamma), tint=float(r.tint), psnr=float(r.psnr), ssim=float(r.ssim), lpips_vgg=float(r.lpips_vgg),
                         dreamsim=float(r.dreamsim))
        for i in VIEWS:
            v = hv[i]
            ref, ctrl = B.opaque("img", sdn, "reference", v, ext=".jpg"), B.opaque("img", sdn, "control", v, ext=".jpg")
            new = B.opaque("img", sdn, "ext:" + name, v, ext=".jpg")
            B.save_jpg(os.path.join(d[0], "test", f"{v:03d}.png"), new)
            man["files"][new] = dict(scene=s, cond="mix5:" + name, view=v)
            man["trials"].append(dict(tid=f"ext/{sdn}/{v}", part="image", kind="ext", scene=s, view=v, ref=ref, a=ctrl, b=new,
                                      a_cond="control", b_cond="mix5:" + name))
        print(s, chosen[s])
    if set(chosen) == have:
        print("nothing to add")
        return
    man["version"] = int(man.get("version", 1)) + 1
    man["ext"] = dict(views=VIEWS, chosen=chosen)
    json.dump(man, open(mf + ".tmp", "w"), indent=1)
    os.replace(mf + ".tmp", mf)
    print(len(man["trials"]), "trials in the manifest;", sum(t["kind"] == "ext" for t in man["trials"]), "ext")
if __name__ == "__main__":
    main()
