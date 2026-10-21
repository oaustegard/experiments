"""Static server for leaf-wasm pages that also accepts uploads from them.

Usage: python3 run/serve.py [port]     (default 8766, binds 127.0.0.1, serves the leaf-wasm root)

GET serves files (symlinks followed, so out/posts.jsonl and out/q can point at the corpus and
queries elsewhere in the repo). PUT /upload/<name> writes the body to out/gpu/<name>, so a
browser page can hand back vectors it computed (web/gpu-bench.html, mode=corpus) for
run/score_vectors.py. Local use only.
"""
import http.server
import re
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT / "out" / "gpu"


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_PUT(self):
        m = re.fullmatch(r"/upload/([A-Za-z0-9_.-]+)", self.path)
        if not m:
            self.send_error(404)
            return
        UP.mkdir(parents=True, exist_ok=True)
        n = int(self.headers["Content-Length"])
        with open(UP / m.group(1), "wb") as f:
            while n:
                chunk = self.rfile.read(min(n, 1 << 20))
                f.write(chunk)
                n -= len(chunk)
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT))).serve_forever()
