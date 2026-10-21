"""Local server for the indexing page: static files, uploads, and an image proxy.

Usage: python3 run/serve.py [port]     (default 8767, binds 127.0.0.1, serves the experiments repo root,
       so pages can import sibling experiments: /bsky-image-search/web/search.html)

GET /img?u=<url> fetches an image from cdn.bsky.app and returns it with CORS allowed: the CDN
sends no Access-Control-Allow-Origin, so a page cannot read its pixels directly. Only that
host is proxied. PUT /upload/<name> writes the body to out/emb/<name>. Local use only.
"""
import http.server
import re
import sys
import urllib.parse
import urllib.request
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT / "out" / "emb"


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/img?"):
            u = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("u", [""])[0]
            if urllib.parse.urlparse(u).hostname != "cdn.bsky.app":
                self.send_error(400, "only cdn.bsky.app is proxied")
                return
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "muninn-experiments"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    body, ctype = r.read(), r.headers.get("Content-Type", "image/jpeg")
            except Exception as e:  # noqa: BLE001 (report the upstream failure to the page)
                self.send_error(502, str(e)[:200])
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

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

    def log_message(self, *a):  # quiet: thousands of /img requests
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8767
    http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT.parent))).serve_forever()
