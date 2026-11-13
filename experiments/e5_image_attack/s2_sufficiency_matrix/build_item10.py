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
REC = P.S2_REC


def include(m):
    key = m.group(1)
    path = os.path.join(REC, key[5:]) if key.startswith("file:") else os.path.join(REC, "results", f"{key}.md")
    return open(path).read().rstrip("\n") if os.path.exists(path) else f"（{key}：尚未生成）"
body = "\n\n".join(open(f).read().rstrip("\n") for f in sorted(glob.glob(os.path.join(REC, "text", "*.md"))))
body = re.sub(r"\{\{([^}]+)\}\}", include, body) + "\n"
open(os.path.join(REC, "item10.md"), "w").write(body)
print("item 10:", body.count("\n"), "lines")
