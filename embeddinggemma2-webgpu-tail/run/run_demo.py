"""Smoke-run web/demo.html in Chromium (SwiftShader WebGPU) with the dummy model.

Usage: python3 -I run/run_demo.py [--pool] [--bench-reps N] [--scan-reps N]

Loads /web/demo.html?local=1&model=dummy (plus &pool=1 with --pool, which forces the
last_hidden_state + tail.pool path), waits for the startup to finish, runs one search,
clicks the bench button, and checks:
  - the startup gate passed,
  - a search returned 10 hits,
  - the bench panel produced numbers,
  - zero page errors (uncaught exceptions and console errors).
Writes out/demo/<variant>.json and exits nonzero if any check fails.
"""
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "demo"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
CHROME_ARGS = [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--use-angle=swiftshader",
    "--use-webgpu-adapter=swiftshader",
    "--enable-features=Vulkan",
]
SEARCH_TEXT = "who won the match in the last minute"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout: float = 20.0) -> None:
    end = time.time() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            if time.time() > end:
                raise SystemExit(f"http.server did not come up for {url}")
            time.sleep(0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true", help="force the last_hidden_state + tail.pool path")
    ap.add_argument("--bench-reps", type=int, default=3)
    ap.add_argument("--scan-reps", type=int, default=2)
    args = ap.parse_args()

    variant = "dummy-pool" if args.pool else "dummy"
    query = f"/web/demo.html?local=1&model=dummy&bench_reps={args.bench_reps}&scan_reps={args.scan_reps}"
    if args.pool:
        query += "&pool=1"

    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    page_errors: list[str] = []
    console_errors: list[str] = []
    summary: dict = {"variant": variant, "url": query}
    try:
        base = f"http://127.0.0.1:{port}"
        wait_for_server(base + "/web/demo.html")
        t0 = time.time()
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
            try:
                page = browser.new_page()
                page.set_default_timeout(0)
                page.on(
                    "console",
                    lambda m: console_errors.append(f"[{m.type}] {m.text}")
                    if m.type == "error" else None,
                )
                page.on("pageerror", lambda e: page_errors.append(str(e)))
                page.goto(base + query, wait_until="load", timeout=60_000)

                page.wait_for_function(
                    "window.demoState && (window.demoState.phase === 'ready' || window.demoState.phase === 'error')",
                    timeout=30 * 60_000, polling=500,
                )
                st = page.evaluate("() => window.demoState")
                summary["startup"] = {
                    "phase": st["phase"], "error": st["error"], "mode": st["mode"],
                    "inputs": st["inputs"], "outputs": st["outputs"], "path": st.get("pathUsed"),
                    "gate": st["gate"],
                }
                if st["phase"] == "ready" and st["gate"] and st["gate"]["passed"]:
                    page.fill("#q", SEARCH_TEXT)
                    page.click("#search-btn")
                    page.wait_for_function(
                        "window.demoState.searches.length >= 1", timeout=10 * 60_000, polling=250,
                    )
                    st = page.evaluate("() => window.demoState")
                    hits = st["searches"][-1]
                    summary["search"] = {
                        "query": hits["query"], "n_hits": len(hits["hits"]),
                        "recall_at_10": hits["recall_at_10"],
                        "top3": [{"doc": h["doc"], "gpu": h["gpu"], "exact": h["exact"]} for h in hits["hits"][:3]],
                    }

                    page.click("#bench-btn")
                    page.wait_for_function(
                        "window.demoState.bench !== null || window.demoState.error !== null",
                        timeout=180 * 60_000, polling=1000,
                    )
                    st = page.evaluate("() => window.demoState")
                    summary["bench"] = st["bench"]
                    summary["bench_error"] = st["error"]
                    summary["bench_panel_text"] = page.inner_text("#bench-out")[:400]
                summary["gate_panel_text"] = page.inner_text("#gate")
                shot = OUT / f"{variant}.png"
                OUT.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(shot), full_page=True)
                summary["screenshot"] = str(shot)
            finally:
                browser.close()
        summary["wall_clock_s"] = round(time.time() - t0, 1)
    finally:
        server.terminate()
        server.wait(timeout=10)

    summary["page_errors"] = page_errors
    summary["console_errors"] = console_errors

    checks = {
        "startup_gate_passed": bool(summary.get("startup", {}).get("gate", {}) and summary["startup"]["gate"]["passed"]),
        "search_returned_10_hits": summary.get("search", {}).get("n_hits") == 10,
        "bench_produced_numbers": bool(summary.get("bench")) and summary["bench"].get("embed_only_ms") is not None
            and summary["bench"].get("scan") is not None,
        "zero_page_errors": not page_errors and not console_errors and not summary.get("bench_error")
            and not summary.get("startup", {}).get("error"),
    }
    summary["checks"] = checks
    summary["PASS"] = all(checks.values())

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{variant}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if summary["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
