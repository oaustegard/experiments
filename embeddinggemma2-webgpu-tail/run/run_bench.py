"""Run web/bench_tail.html in Chromium (SwiftShader WebGPU) and write results/bench_tail_swiftshader.json.

Usage: python3 -I run/run_bench.py [--n 100000] [--n-enc 16384] [--reps 3] [--untiled out/tail_untiled.js]

--untiled times an older tail.js's encode next to the current one (path relative to the project
root). SwiftShader is a software rasterizer: these are not GPU timings. On a machine with a GPU,
open /web/bench_tail.html in a browser and read the log instead.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gpu import CHROME, CHROME_ARGS, free_port, wait_for_server  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--n-enc", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--untiled")
    a = ap.parse_args()
    q = f"?n={a.n}&n_enc={a.n_enc}&reps={a.reps}" + (f"&untiled=/{a.untiled}" if a.untiled else "")
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(port)
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
            try:
                page = browser.new_page()
                page.set_default_timeout(0)
                page.goto(f"http://127.0.0.1:{port}/web/bench_tail.html{q}", wait_until="load", timeout=60_000)
                page.wait_for_function("window.benchReady === true", timeout=60_000)
                result = page.evaluate("() => window.runBench()")
            finally:
                browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    result["note"] = "SwiftShader (software) WebGPU on this machine's CPU: not GPU timings."
    out = ROOT / "results" / "bench_tail_swiftshader.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
