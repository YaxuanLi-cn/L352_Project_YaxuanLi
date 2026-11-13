#!/usr/bin/env python
import glob
import json
import os
import sys
import re
import time
import colour as C
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
REC = P.S4_REC
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(P.S4_RUNS, "*", "*", "done.json")))]
started = sorted(d for d in glob.glob(os.path.join(P.S4_RUNS, "*", "*")) if os.path.isdir(d))
unfinished = [d for d in started if not os.path.exists(os.path.join(d, "done.json"))]
log = open(os.path.join(REC, "logs", "queue.log")).read()
per = {s: sorted(j["mult"] for j in runs if j["scene"] == s and j["status"] == "ok") for s in C.SCENES}
full = [s for s in C.SCENES if len(per[s]) == 3]
part = [s for s in C.SCENES if 0 < len(per[s]) < 3]
none = [s for s in C.SCENES if not per[s]]
starts = re.findall(r"^(\d\d:\d\d:\d\d) start ", log, re.M)
ends = re.findall(r"^(\d\d:\d\d:\d\d) end ", log, re.M)
cut = re.findall(r"cut: (\S+) and every scene after it", log)
killed = "watchdog: GPU deadline reached" in log
by_user = "stopped at the user's request" in log
L = [f"- **新跑的运行**：{len(runs)} 次，全部正常结束（" + "；".join(f"{C.NAME[s]} {len(per[s])} 档" for s in C.SCENES if per[s]) + "）。"
     if all(j["status"] == "ok" for j in runs) else f"- **新跑的运行**：{len(runs)} 次，其中 {sum(j['status'] != 'ok' for j in runs)} 次发散。",
     f"- **三档都跑完的场景**：{'、'.join(C.NAME[s] for s in full) or '无'}。"]
if part:
    L.append(f"- **只跑完部分档的场景**（已完成的档列在表里）：{'、'.join(C.NAME[s] + ' ' + str(len(per[s])) + ' 档' for s in part)}。")
if none:
    why = (f"队列在 {C.NAME.get(cut[0], cut[0])} 处判断三档无法在 GPU 截止时间前跑完，把它和排在后面的场景整个砍掉" if cut else
           ("停止时还没有排到或没有跑完" if by_user else "没有排到"))
    L.append(f"- **没有新跑的场景**：{'、'.join(C.NAME[s] for s in none)}（{why}；" +
             "、".join(C.NAME[s] for s in none if s in ("mip360/bonsai", "mip360/garden")) + " 在表里只有第 10 条已有的档）。" if any(s in ("mip360/bonsai", "mip360/garden") for s in none)
             else f"- **没有新跑的场景**：{'、'.join(C.NAME[s] for s in none)}（{why}）。")
if unfinished:
    L.append(f"- {'应你的要求停止时' if by_user else '到点时'}还没结束、被中止的运行 {len(unfinished)} 次（目录里没有 done.json，不计入结果）：" + "、".join(os.path.relpath(d, os.path.dirname(P.S4_RUNS)) for d in unfinished) + "。")
L.append(f"- **GPU 用时**：第一个任务 {starts[0]} UTC 启动，最后一个 {ends[-1]} UTC 结束" + ("（看门狗到点停掉了还在跑的任务）" if killed else "") + ("；应你的要求全部停止" if by_user else "") + f"；训练用时合计 {sum(j['train_time'] for j in runs) / 60:.0f} 分钟（并行）。")
open(os.path.join(REC, "results", "stop.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
