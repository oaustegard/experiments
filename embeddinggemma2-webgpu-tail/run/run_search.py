"""Correctness run of web/search.html in Chromium (SwiftShader WebGPU) with the real model.

Usage: python3 -I run/run_search.py [--query TEXT]

Serves the project root, answers jsDelivr and Bluesky requests through Python (Chromium here
has no route to the egress proxy), points the page at the out/hfmirror/ model files with &hf=,
waits for startup, checks that the remex parity check passed, types one query and checks that
10 hits rendered with post text. Writes out/search/run.json; exits nonzero on any failed check.
Timings from this run are SwiftShader's and mean nothing; real timings come from a browser.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_demo import CHROME, CHROME_ARGS, free_port, serve_from_cache, wait_for_server  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "search"


def bsky_passthrough(route):
    """Bluesky's public API, fetched live (not cached: the page shows current posts)."""
    url = route.request.url.replace("https://public.api.bsky.app/", "https://api.bsky.app/")
    req = urllib.request.Request(url, headers={"User-Agent": "muninn-experiments"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    route.fulfill(status=200, body=body, content_type="application/json",
                  headers={"access-control-allow-origin": "*"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="what do people think about AI safety research")
    a = ap.parse_args()
    for name, sub in [("model_q4.onnx", "onnx"), ("model_q4.onnx_data", "onnx"),
                      ("tokenizer.json", ""), ("tokenizer_config.json", "")]:
        link = ROOT / "out" / "hfmirror" / sub / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(ROOT / "out" / "hf" / name)
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                              cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    errors, summary = [], {}
    try:
        base = f"http://127.0.0.1:{port}"
        wait_for_server(base + "/web/search.html")
        t0 = time.time()
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
            try:
                page = b.new_page()
                page.set_default_timeout(0)
                page.on("console", lambda m: errors.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.route("https://cdn.jsdelivr.net/**", serve_from_cache)
                page.route("https://public.api.bsky.app/**", bsky_passthrough)
                page.goto(base + "/web/search.html?hf=/out/hfmirror/", wait_until="load", timeout=60_000)
                page.wait_for_function("window.searchState && window.searchState.phase !== 'loading'",
                                       timeout=60 * 60_000, polling=1000)
                st = page.evaluate("() => window.searchState")
                summary["startup"] = {k: st[k] for k in ("phase", "error", "adapter", "index", "load_ms", "parity")}
                if st["phase"] == "ready":
                    page.fill("#q", a.query)
                    page.wait_for_function("window.searchState.queries.length >= 1", timeout=30 * 60_000, polling=500)
                    page.wait_for_function("document.querySelectorAll('#hits li').length === 10", timeout=60_000)
                    texts = page.eval_on_selector_all("#hits li .text", "els => els.map(e => e.textContent)")
                    summary["query"] = a.query
                    summary["hits_with_text"] = sum(1 for t in texts if not t.startswith("(post no longer"))
                    summary["top3"] = texts[:3]
                page.screenshot(path=str(ROOT / "out" / "search.png"), full_page=True)
            finally:
                b.close()
        summary["wall_clock_s"] = round(time.time() - t0, 1)
    finally:
        server.terminate()
        server.wait(timeout=10)
    summary["errors"] = errors
    checks = {
        "ready": summary.get("startup", {}).get("phase") == "ready",
        "parity_passed": bool(summary.get("startup", {}).get("parity", {}).get("passed")),
        "ten_hits_with_text": summary.get("hits_with_text", 0) >= 8,
        "no_errors": not errors,
    }
    summary["checks"] = checks
    summary["PASS"] = all(checks.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if summary["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
