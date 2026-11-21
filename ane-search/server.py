#!/usr/bin/env python3
"""Local semantic search over the index. Binds to 127.0.0.1 only.
GET /            the page
GET /search?q=   JSON results (semantic + BM25, fused by reciprocal rank)
GET /open?i=     open that result's file in the default app (only paths that are in the index)"""
import os, sys, json, time, math, subprocess, urllib.parse, numpy as np, warnings
warnings.filterwarnings("ignore")
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from common import *
PORT = int(os.environ.get("PORT", "8765"))
def load_index():
    """(Re)load the knowledge base; called at start and whenever index.py has written a newer stats.json."""
    global KB, META, STATS, N, STAMP
    import kb
    STAMP = os.path.getmtime(f"{IDX}/stats.json")
    KB = kb.Reader(); META = KB.meta; N = int(KB.live.sum())
    STATS = json.load(open(f"{IDX}/stats.json"))
load_index()
def fresh():
    try:
        if os.path.getmtime(f"{IDX}/stats.json") != STAMP: load_index()
    except Exception: pass                       # mid-write: keep serving the index already in memory
EMB = Embedder(docs=False)

def search(q, k=20, mode="hybrid", under=""):
    t0 = time.perf_counter(); qv = EMB.query(q); t1 = time.perf_counter()
    sem = KB.semantic(qv); lex = KB.keyword(q)
    if under:
        keep = np.fromiter((m["path"].startswith(under) for m in META), bool, len(META)); sem = np.where(keep, sem, -1); lex = np.where(keep, lex, 0)
    depth = 200
    top_of = lambda a: [int(i) for i in np.lexsort((np.arange(len(a)), -a))[:depth]]         # ties broken by row, as the app does
    rs = [i for i in top_of(sem) if sem[i] > -1]; rl = [i for i in top_of(lex) if lex[i] > 0]
    score = {}
    if mode in ("hybrid", "semantic"):
        for r, i in enumerate(rs): score[i] = score.get(i, 0) + 1 / (60 + r)
    if mode in ("hybrid", "keyword"):
        for r, i in enumerate(rl): score[i] = score.get(i, 0) + 1 / (60 + r)
    top, per_file, out = sorted(score, key=lambda i: (-score[i], i)), {}, []
    for i in top:
        m = META[i]
        if per_file.get(m["path"], 0) >= 2: continue                # at most two chunks per file
        per_file[m["path"]] = per_file.get(m["path"], 0) + 1
        out.append({"i": i, "path": m["path"], "head": m["head"], "line": m["line"], "text": m["text"][:700], "cos": round(float(sem[i]), 3), "bm25": round(float(lex[i]), 1)})
        if len(out) >= k: break
    t2 = time.perf_counter()
    return {"q": q, "results": out, "embed_ms": round((t1 - t0) * 1000, 2), "rank_ms": round((t2 - t1) * 1000, 2), "chunks": N}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, body, ctype="application/json", code=200):
        body = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); qs = urllib.parse.parse_qs(u.query)
        fresh()
        try:
            if u.path == "/": return self.send(open(os.path.join(HERE, "index.html"), "rb").read(), "text/html")
            if u.path == "/stats": return self.send(json.dumps({**STATS, "root": ROOT}))
            if u.path == "/search":
                q = (qs.get("q") or [""])[0].strip()
                if not q: return self.send(json.dumps({"q": "", "results": [], "chunks": N}))
                return self.send(json.dumps(search(q, int((qs.get("k") or ["20"])[0]), (qs.get("mode") or ["hybrid"])[0], (qs.get("under") or [""])[0])))
            if u.path == "/open":
                m = META[int(qs["i"][0])]                               # index lookup: the client never supplies a path
                subprocess.Popen(["open", os.path.join(ROOT, m["path"])]); return self.send(json.dumps({"opened": m["path"]}))
            self.send("not found", "text/plain", 404)
        except Exception as e:
            self.send(json.dumps({"error": f"{type(e).__name__}: {e}"}), code=500)

if __name__ == "__main__":
    if len(sys.argv) > 1:                                               # CLI: python server.py "query"
        r = search(" ".join(sys.argv[1:]), 8)
        print(f"embed {r['embed_ms']} ms, rank {r['rank_ms']} ms over {r['chunks']} chunks")
        for x in r["results"]: print(f"  {x['cos']:.3f} {x['bm25']:5.1f}  {x['path']}:{x['line']}  [{x['head'][:60]}]  {x['text'][:90]!r}")
    else:
        print(f"searching {N} chunks on http://127.0.0.1:{PORT}", flush=True)
        ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
