#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "code", "common", "paths.py")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "code", "common"))
import paths as P              
import helpers as C              
import yaml              
PY = sys.executable
ORIG_CODE = P.code(P.E4, "original_run")


def step_analyse(runs_root, out):
    cfg = yaml.safe_load(open(os.path.join(ORIG_CODE, "configs", "full.yaml")))
    cfg["runs_root"] = P.rel(runs_root)
    cfg["results_dir"] = P.rel(os.path.join(out, "results"))
    f = os.path.join(out, "analyse_config.yaml")
    yaml.safe_dump(cfg, open(f, "w"), sort_keys=False)
    C.log(f"analyse.py on {runs_root}")
    subprocess.run([PY, os.path.join(ORIG_CODE, "analyse.py"), f], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=P.RERUN_RUNS)
    ap.add_argument("--out", default=P.RERUN_WHOLE)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    step_analyse(os.path.abspath(args.runs_root), out)
if __name__ == "__main__":
    main()
