#!/usr/bin/env python3
import argparse
import json
import os
import sys
import random
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              


def base_cond(r):
    if r["kind"] in ("attack", "floor", "ext"):
        return "control"
    if r["kind"] == "check":
        return "reference-copy"
    return r["A_cond"] if not r["A_cond"].startswith("mix") else r["B_cond"]


def share(rows):
    return (sum(r["ok"] for r in rows) / len(rows)) if rows else float("nan")


def boot(rows, pids, n=2000, seed=0):
    by = {}
    for r in rows:
        by.setdefault(r["pid"], []).append(r["ok"])
    ps = [p for p in pids if p in by]
    if len(ps) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = tot = 0
        for p in (rng.choice(ps) for _ in ps):
            k += sum(by[p])
            tot += len(by[p])
        out.append(k / tot)
    out.sort()
    return (out[int(0.025 * n)], out[int(0.975 * n) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=P.STUDY_DATA)
    args = ap.parse_args()
    rows, sessions = [], {}
    for f in sorted(os.listdir(os.path.join(args.data, "sessions"))):
        s = json.load(open(os.path.join(args.data, "sessions", f)))
        sessions[s["pid"]] = s
    for f in sorted(os.listdir(os.path.join(args.data, "responses"))):
        seen = set()
        for l in open(os.path.join(args.data, "responses", f)):
            r = json.loads(l)
            if (r["pid"], r["i"]) in seen:
                continue
            seen.add((r["pid"], r["i"]))
            r["ok"] = int(r["chosen"] == base_cond(r))
            rows.append(r)
    pids = sorted(set(r["pid"] for r in rows))
    checks = {p: [r["ok"] for r in rows if r["pid"] == p and r["kind"] == "check"] for p in pids}
    excluded = [p for p in pids if any(v == 0 for v in checks[p])]
    valid = [p for p in pids if p not in excluded]
    finished = [p for p in valid if len([r for r in rows if r["pid"] == p]) >= len(sessions[p]["trials"])]
    use = [r for r in rows if r["pid"] in valid and r["rep"] == 0]
    scenes = ["gsqa/truck", "gsqa/flowers", "gsqa/playground2", "gsqa/train"]
    res = dict(n_started=len(sessions), n_with_answers=len(pids), n_excluded=len(excluded), n_valid=len(valid), n_finished=len(finished),
               shares={})
    L = [f"# Web 2AFC study: summary", "",
         f"Sessions started {len(sessions)}, with answers {len(pids)}, excluded (attention check) {len(excluded)}, valid {len(valid)}, "
         f"valid and finished {len(finished)}.", "",
         "| kind | scene | answers | share choosing control / original | 95 % CI (participants resampled) |", "|:--|:--|--:|--:|:--|"]
    for kind in ("attack", "floor", "ext", "flicker", "vfloor", "check"):
        groups = [(s, [r for r in use if r["kind"] == kind and r["scene"] == s]) for s in scenes]
        if kind == "attack":
            groups.append(("three circumvented scenes", [r for r in use if r["kind"] == kind and r["scene"] != "gsqa/train"]))
        if kind in ("flicker", "vfloor", "floor", "ext"):
            groups.append(("all scenes", [r for r in use if r["kind"] == kind]))
        if kind == "check":
            groups = [("all", [r for r in rows if r["kind"] == "check"])]
        for name, g in groups:
            if not g:
                continue
            lo, hi = boot(g, valid)
            res["shares"][f"{kind}|{name}"] = dict(n=len(g), share=share(g), lo=lo, hi=hi)
            L.append(f"| {kind} | {name.split('/')[-1]} | {len(g)} | {share(g):.2f} | [{lo:.2f}, {hi:.2f}] |")
    ext = [r for r in use if r["kind"] == "ext"]
    ext_p = sorted(set(r["pid"] for r in ext))
    ext_fin = [p for p in ext_p if p in finished]
    res["n_ext_participants"], res["n_ext_finished"] = len(ext_p), len(ext_fin)
    if ext:
        views = set((r["scene"], r["tid"].split("/")[-1]) for r in ext)
        same = [r for r in use if r["kind"] == "attack" and r["pid"] in ext_p and (r["scene"], r["tid"].split("/")[-1]) in views]
        fl = [r for r in use if r["kind"] == "floor" and r["pid"] in ext_p]
        for name, g in (("ext|participants with ext trials", ext), ("attack|same views, participants with ext trials", same),
                        ("floor|participants with ext trials", fl)):
            if g:
                lo, hi = boot(g, valid)
                res["shares"][name] = dict(n=len(g), share=share(g), lo=lo, hi=hi)
        L += ["", f"Participants with `ext` trials: {len(ext_p)} (finished {len(ext_fin)}). Control chosen: ext {share(ext):.2f} of {len(ext)}; "
                  f"attack on the same views {share(same):.2f} of {len(same)}; floor {share(fl):.2f} of {len(fl)}."]
    rep = [r for r in rows if r["pid"] in valid and r["rep"] == 1]
    same = n = 0
    for r in rep:
        first = [q for q in rows if q["pid"] == r["pid"] and q["tid"] == r["tid"] and q["rep"] == 0]
        n += 1
        same += int(first[0]["chosen"] == r["chosen"])
    res["consistency"] = dict(n=n, share=(same / n if n else float("nan")))
    L += ["", f"Repeated trials: same model chosen in {same} of {n}."]
    for lab, cond in (("displayed width < 700 px", lambda r: (r.get("shown_w") or 0) < 700), ("displayed width >= 700 px", lambda r: (r.get("shown_w") or 0) >= 700)):
        g = [r for r in use if r["kind"] == "attack" and r["scene"] != "gsqa/train" and cond(r)]
        if g:
            L.append(f"Attack trials (three circumvented scenes), {lab}: {share(g):.2f} of {len(g)} answers choose the control.")
            res["shares"][f"attack|{lab}"] = dict(n=len(g), share=share(g))
    os.makedirs(P.STUDY_RESULTS, exist_ok=True)
    open(os.path.join(P.STUDY_RESULTS, "summary.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(P.STUDY_RESULTS, "summary.json"), "w"), indent=1)
    print("\n".join(L))
if __name__ == "__main__":
    main()
