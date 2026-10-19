"""Run web/test_topk.html in Chromium (SwiftShader WebGPU): GPU top-k against the host top-k.

Usage: python3 -I run/run_topk_edge.py

Cases cover n < k, n around the 1024-pair chunk, k = 1 and k = 256, a three-pass n, and
duplicated rows (identical scores, so the lower-index tie rule decides). Writes out/topk_edge.json
and exits nonzero on any mismatch, short result or WebGPU error.
"""
import json
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gpu import CHROME, CHROME_ARGS, free_port  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        from run_gpu import wait_for_server
        wait_for_server(port)
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
            try:
                page = browser.new_page()
                page.set_default_timeout(0)
                page.goto(f"http://127.0.0.1:{port}/web/test_topk.html", wait_until="load", timeout=60_000)
                page.wait_for_function("window.topkReady === true", timeout=60_000)
                result = page.evaluate("() => window.runTopk()")
            finally:
                browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    ok = not result["errors"] and all(
        c["mismatches"] == 0 and c["returned"] == c["expected"] for c in result["cases"])
    result["PASS"] = ok
    out = ROOT / "out" / "topk_edge.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
