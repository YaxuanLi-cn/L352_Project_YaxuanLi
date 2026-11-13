#!/usr/bin/env python3
import argparse
import csv
import io
import json
import os
import random
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()
PID_RE = re.compile(r"^[a-f0-9]{16}$")
STIM_RE = re.compile(r"^[a-f0-9]{14}\.(jpg|mp4)$")
N_REPEAT = 2


class Study:
    def __init__(self, data):
        self.data = data
        for d in ("sessions", "responses"):
            os.makedirs(os.path.join(data, d), exist_ok=True)
        kf = os.path.join(data, "keys.json")
        if not os.path.exists(kf):
            json.dump(dict(study=secrets.token_hex(4), admin=secrets.token_hex(10)), open(kf, "w"))
        self.keys = json.load(open(kf))
    def manifest(self):
        return json.load(open(os.path.join(self.data, "manifest.json")))
    def index(self):
        return open(os.path.join(HERE, "index.html"), "rb").read()
    def session_file(self, pid):
        return os.path.join(self.data, "sessions", f"{pid}.json")
    def response_file(self, pid):
        return os.path.join(self.data, "responses", f"{pid}.jsonl")
    def answers(self, pid):
        f = self.response_file(pid)
        return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
    def new_session(self, info, ip):
        man = self.manifest()
        pid = secrets.token_hex(8)
        rng = random.Random(pid)
        def sides(n):
            x = [True, False] * (n // 2) + [rng.random() < 0.5] * (n % 2)
            rng.shuffle(x)
            return x
        def place(t, a_first, rep=0):
            return dict(tid=t["tid"], part=t["part"], kind=t["kind"], scene=t["scene"], rep=rep, ref=t["ref"],
                        A=t["a"] if a_first else t["b"], B=t["b"] if a_first else t["a"],
                        A_cond=t["a_cond"] if a_first else t["b_cond"], B_cond=t["b_cond"] if a_first else t["a_cond"])
        img = [t for t in man["trials"] if t["part"] == "image"]
        image = [place(t, s) for t, s in zip(img, sides(len(img)))]
        rng.shuffle(image)
        firsts = rng.sample([t for t in image if t["kind"] == "attack"], N_REPEAT)
        for first in firsts:
            rep = dict(first, rep=1, A=first["B"], B=first["A"], A_cond=first["B_cond"], B_cond=first["A_cond"])
            i = next(k for k, t in enumerate(image) if t is first)
            image.insert(rng.randint(min(len(image), i + 8), len(image)), rep)
        if image[0]["kind"] != "attack":
            j = next(k for k, t in enumerate(image) if t["kind"] == "attack" and t["rep"] == 0)
            image[0], image[j] = image[j], image[0]
        groups = {}
        for t in man["trials"]:
            if t["part"] == "video":
                groups.setdefault(t["group"], []).append(t)
        vid = [rng.choice(g) for g in groups.values()]
        video = [place(t, s) for t, s in zip(vid, sides(len(vid)))]
        rng.shuffle(video)
        trials = image + video
        s = dict(pid=pid, created=time.time(), ip=ip, info=info, manifest_version=man.get("version"), trials=trials)
        json.dump(s, open(self.session_file(pid), "w"), indent=1)
        return s
    def client_trials(self, s):
        return [dict(i=i, part=t["part"], ref=t["ref"], A=t["A"], B=t["B"]) for i, t in enumerate(s["trials"])]
    def record(self, pid, body):
        s = json.load(open(self.session_file(pid)))
        i = int(body["i"])
        t = s["trials"][i]
        choice = body["choice"]
        assert choice in ("A", "B")
        row = dict(pid=pid, i=i, tid=t["tid"], part=t["part"], kind=t["kind"], scene=t["scene"], rep=t["rep"], A_cond=t["A_cond"],
                   B_cond=t["B_cond"], choice=choice, chosen=t[choice + "_cond"], t=time.time())
        for k in ("rt", "views", "switches", "vw", "vh", "dpr", "shown_w", "hidden", "first"):
            if k in body:
                row[k] = body[k]
        with LOCK:
            with open(self.response_file(pid), "a") as f:
                f.write(json.dumps(row) + "\n")
        return row
    def all_rows(self):
        rows = []
        for f in sorted(os.listdir(os.path.join(self.data, "responses"))):
            rows += [json.loads(l) for l in open(os.path.join(self.data, "responses", f))]
        return rows
    def summary_html(self):
        sess = [json.load(open(os.path.join(self.data, "sessions", f))) for f in sorted(os.listdir(os.path.join(self.data, "sessions")))]
        rows = self.all_rows()
        done = {}
        for r in rows:
            done.setdefault(r["pid"], set()).add(r["i"])
        n_full = sum(1 for s in sess if len(done.get(s["pid"], ())) >= len(s["trials"]))
        agg = {}
        for r in rows:
            base = {"attack": "control", "floor": "control", "check": "reference-copy"}.get(r["kind"])
            if base is None:
                base = r["A_cond"] if not r["A_cond"].startswith("mix") else r["B_cond"]
            k = (r["kind"], r["scene"])
            a = agg.setdefault(k, [0, 0])
            a[0] += 1
            a[1] += int(r["chosen"] == base)
        h = ["<html><meta charset='utf-8'><body style='font-family:sans-serif'>",
             f"<h3>Sessions started: {len(sess)} &nbsp; finished: {n_full} &nbsp; answers: {len(rows)}</h3>",
             "<p><a href='responses.csv'>responses.csv</a></p>",
             "<table border=1 cellpadding=4 style='border-collapse:collapse'><tr><th>kind</th><th>scene</th><th>answers</th>"
             "<th>chose control / original / reference</th><th>share</th></tr>"]
        for (kind, scene), (n, k) in sorted(agg.items()):
            h.append(f"<tr><td>{kind}</td><td>{scene}</td><td>{n}</td><td>{k}</td><td>{k / n:.2f}</td></tr>")
        h.append("</table><p>attack: control vs attacked model; floor: control vs model trained with the triplet loss without attack; "
                 "check: reference copy vs control; flicker / vfloor: original video vs mixture (sparse views / fully trained).</p>")
        h.append("<table border=1 cellpadding=4 style='border-collapse:collapse'><tr><th>pid</th><th>started (UTC)</th><th>answers</th><th>device</th></tr>")
        for s in sess:
            ua = (s.get("info") or {}).get("ua", "")[:90]
            h.append(f"<tr><td>{s['pid']}</td><td>{time.strftime('%m-%d %H:%M', time.gmtime(s['created']))}</td>"
                     f"<td>{len(done.get(s['pid'], ()))} / {len(s['trials'])}</td><td>{ua}</td></tr>")
        h.append("</table></body></html>")
        return "\n".join(h).encode()
    def csv_bytes(self):
        rows = self.all_rows()
        cols = ["pid", "i", "tid", "part", "kind", "scene", "rep", "A_cond", "B_cond", "choice", "chosen", "rt", "switches", "views",
                "vw", "vh", "dpr", "shown_w", "hidden", "first", "t"]
        buf = io.StringIO()
        w = csv.DictWriter(buf, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
        return buf.getvalue().encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    study = None
    def log_message(self, fmt, *args):
        pass
    def send(self, code, body=b"", ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
    def serve_file(self, path, ctype):
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send(416, b"", extra={"Content-Range": f"bytes */{size}"})
                    return
                code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(262144, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)
    def do_HEAD(self):
        self.do_GET()
    def do_GET(self):
        st = self.study
        p = self.path.split("?")[0]
        try:
            if p in (f"/s/{st.keys['study']}", f"/s/{st.keys['study']}/"):
                return self.send(200, st.index(), "text/html; charset=utf-8", {"Cache-Control": "no-store"})
            m = re.match(r"^/stim/([a-f0-9]{14}\.(?:jpg|mp4))$", p)
            if m:
                name = m.group(1)
                sub, ctype = ("img", "image/jpeg") if name.endswith(".jpg") else ("vid", "video/mp4")
                f = os.path.join(st.data, sub, name)
                if os.path.exists(f):
                    return self.serve_file(f, ctype)
                return self.send(404, b"{}")
            if p in (f"/admin/{st.keys['admin']}", f"/admin/{st.keys['admin']}/"):
                return self.send(200, st.summary_html(), "text/html; charset=utf-8", {"Cache-Control": "no-store"})
            if p == f"/admin/{st.keys['admin']}/responses.csv":
                return self.send(200, st.csv_bytes(), "text/csv; charset=utf-8", {"Cache-Control": "no-store"})
            self.send(404, b"{}")
        except Exception as e:                
            self.send(500, json.dumps(dict(error=str(e))).encode())
    def do_POST(self):
        st = self.study
        p = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(min(n, 200000)) or b"{}")
            if p == f"/api/{st.keys['study']}/start":
                with LOCK:
                    s = st.new_session(body, self.headers.get("X-Forwarded-For") or self.client_address[0])
                return self.send(200, json.dumps(dict(pid=s["pid"], trials=st.client_trials(s))).encode())
            if p == f"/api/{st.keys['study']}/answer":
                pid = body.get("pid", "")
                if not PID_RE.match(pid) or not os.path.exists(st.session_file(pid)):
                    return self.send(400, b'{"error":"pid"}')
                st.record(pid, body)
                return self.send(200, b'{"ok":true}')
            if p == f"/api/{st.keys['study']}/done":
                pid = body.get("pid", "")
                if PID_RE.match(pid) and os.path.exists(st.session_file(pid)):
                    with LOCK:
                        s = json.load(open(st.session_file(pid)))
                        s["finished"] = time.time()
                        s["exit"] = {k: str(body.get(k, ""))[:2000] for k in ("comment", "device", "vision")}
                        json.dump(s, open(st.session_file(pid), "w"), indent=1)
                return self.send(200, b'{"ok":true}')
            self.send(404, b"{}")
        except Exception as e:                
            self.send(500, json.dumps(dict(error=str(e))).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=10200)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "..", "data", "experiments", "e6_user_study", "study_data"))
    args = ap.parse_args()
    Handler.study = Study(args.data)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.daemon_threads = True
    print(f"study key {Handler.study.keys['study']}  admin key {Handler.study.keys['admin']}  port {args.port}", flush=True)
    srv.serve_forever()
if __name__ == "__main__":
    main()
