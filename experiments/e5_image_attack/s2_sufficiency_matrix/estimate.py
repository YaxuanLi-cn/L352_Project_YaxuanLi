#!/usr/bin/env python
import glob
import json
import os
import sys
import time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
start, deadline = int(open(os.path.join(P.S2_REC, "gpu_start.txt")).read()), int(open(os.path.join(P.S2_REC, "deadline.txt")).read())
now = int(time.time())
jobs = [l.split() for l in open(os.path.join(HERE, "jobs.txt")) if l.strip() and not l.startswith("#")]
done = {}
for f in glob.glob(os.path.join(P.S2_RUNS, "*", "*", "done.json")):
    j = json.load(open(f))
    done[(j["scene"], j["loss"], j["seed"])] = j
d = pd.read_csv(os.path.join(P.RERUN_RESULTS, "metrics.csv"))
tt = d[(d.split == "model") & (d.seed == 0) & (d.metric == "train_time_s")].set_index(["scene", "loss", "variant"]).value
ratio = {s: float(np.median([tt[(s, l, v)] / tt[("gsqa/truck", l, v)] for l, v in (("default", "ctrl"), ("lpips_vgg", "mixed"), ("dists", "mixed"))]))
         for s in ("gsqa/truck", "mip360/bonsai", "mip360/garden")}
per = {}
for s in ratio:
    w = [j["wall_time"] for k, j in done.items() if k[0] == s]
    per[s] = float(np.mean(w)) if w else None
base = per["gsqa/truck"]
est = {s: (per[s] if per[s] else base * ratio[s]) for s in ratio}
left = [j for j in jobs if (j[1], j[2], int(j[3])) not in done]
seq_left = sum(est[j[1]] for j in left)
seq_all = sum(est[j[1]] for j in jobs)
spent = now - start
gain = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
L = [f"- 计时时刻：{time.strftime('%H:%M:%S', time.gmtime(now))} UTC；GPU 已用 {spent / 60:.0f} 分钟（含 A 组和读数校准），剩余 {(deadline - now) / 60:.0f} 分钟。",
     f"- 已完成 {len(done)} / {len(jobs)} 次训练。每次训练加评测的进程用时：" +
     "；".join(f"{s.split('/')[1]} {est[s] / 60:.1f} 分钟（{'实测 ' + str(sum(1 for k in done if k[0] == s)) + ' 次的均值' if per[s] else '按重跑实验的训练用时之比 ' + format(ratio[s], '.2f') + ' 由 truck 推算'}）" for s in ratio) + "。",
     f"- 整份清单（{len(jobs)} 次）逐个顺序跑需要 {seq_all / 60:.0f} 分钟；剩下的 {len(left)} 次顺序跑需要 {seq_left / 60:.0f} 分钟"
     f"{'，超出' if seq_left > deadline - now else '，不超出'}剩余预算。",
     f"- 按并行吞吐量 {gain:.2f} 倍估计：剩下的需要 {seq_left / gain / 60:.0f} 分钟，预计 {time.strftime('%H:%M', time.gmtime(now + seq_left / gain))} UTC 完成，"
     f"{'**超出**' if seq_left / gain > deadline - now else '在'}截止时间（{time.strftime('%H:%M', time.gmtime(deadline))} UTC）{'' if seq_left / gain > deadline - now else '之前'}。"]
open(os.path.join(P.S2_REC, "results", "estimate.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
json.dump(dict(now=now, per_scene_seconds=est, ratio=ratio, seq_left=seq_left, seq_all=seq_all, gain=gain, n_done=len(done)), open(os.path.join(P.S2_REC, "results", f"estimate_{now}.json"), "w"))
