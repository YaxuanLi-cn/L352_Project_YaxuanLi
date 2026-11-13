#!/usr/bin/env python
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
PAGE = P.rec(P.E5, P.S3, "page")                                                                    
t = open(os.path.join(HERE, "template.html")).read()
trials = json.load(open(os.path.join(PAGE, "trials.json")))
t = t.replace("/*TRIALS*/[]", json.dumps(trials, ensure_ascii=False, separators=(",", ":")))
lad = os.path.join(PAGE, "ladder.json")
if "/*LADDER*/null" in t:
    t = t.replace("/*LADDER*/null", json.dumps(json.load(open(lad)), ensure_ascii=False, separators=(",", ":")) if os.path.exists(lad) else "null")
open(os.path.join(PAGE, "index.html"), "w").write(t)
files = sorted(f"img/{f}" for f in os.listdir(os.path.join(PAGE, "img")))
json.dump(files, open(os.path.join(PAGE, "files.json"), "w"))
print(len(trials), "trials;", len(files), "image files;", round(sum(os.path.getsize(os.path.join(PAGE, f)) for f in files) / 1e6, 1), "MB; html", os.path.getsize(os.path.join(PAGE, "index.html")), "bytes")
