#!/usr/bin/env python
import glob, json, os, re, sys, time
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
L = P.PILOT_LOGS
since = sys.argv[1] if len(sys.argv) > 1 else '00:00:00'
tot, n = 0.0, 0
for f in sorted(glob.glob(L + '/rerun_g*.log')):
    cur, steps = None, []
    for line in open(f, errors='ignore'):
        m = re.search(r'START (\S+) s(\d) (\S+)/(\S+)', line)
        if m:
            cur, steps = m.groups(), []
        m = re.search(r'(\d\d:\d\d:\d\d)\s+step (\d+)/10000 loss \S+ (\d+)s', line)
        if m:
            steps.append((m.group(1), int(m.group(2)), int(m.group(3))))
    name = os.path.basename(f)[6:-4]
    fresh = [s for s in steps if s[0] >= since or s[0] < '12:00:00' <= since]
    if not cur or len(steps) < 2 or not fresh:
        print(f"{name:4s} {str(cur):50s} no new step line since {since}")
        continue
    ms = (steps[-1][2] - steps[-2][2]) / (steps[-1][1] - steps[-2][1]) * 1000
    sc, sd, loss, var = cur
    oms = json.load(open(os.path.join(P.ORIG_RUNS, sc.replace('/', '_'), f"s{sd}", f"{loss}__{var}", "done.json")))['train_time'] / 10
    tot += oms / ms; n += 1
    print(f"{name:4s} {str(cur):50s} {ms:6.0f} ms/step, original {oms:5.0f}, x{ms / oms:4.2f}  (line at {steps[-1][0]})")
print(f'{n} workers measured; sum of relative speeds = {tot:.2f}; mean {tot / max(n, 1):.2f} per worker')
