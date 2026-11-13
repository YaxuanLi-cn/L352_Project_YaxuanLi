#!/usr/bin/env python
import glob
import os
import re
import sys
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
REC = P.S4_REC


def include(m):
    path = os.path.join(REC, m.group(1))
    return open(path).read().rstrip("\n") if os.path.exists(path) else f"（{m.group(1)}：尚未生成）"
body = "\n\n".join(open(f).read().rstrip("\n") for f in sorted(glob.glob(os.path.join(REC, "text", "*.md"))))
body = re.sub(r"\{\{file:([^}]+)\}\}", include, body) + "\n"
open(os.path.join(REC, "item11.md"), "w").write(body)
print("item 11:", body.count("\n"), "lines")
