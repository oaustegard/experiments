"""Run web/index.html (the bench) in headless Chromium (SwiftShader WebGPU) and check the engine's vectors.

Usage: python3 -I run/check.py [--reps 2] [--ort]

Serves the experiments repo, answers jsDelivr from a disk cache (headless Chromium has no route
to the egress proxy), opens index.html?auto=1 and waits for it. Passes when every WGSL vector,
single and batched, is at cosine >= 0.9999 to onnxruntime CPU. SwiftShader timings are recorded
in results/check_swiftshader.json and mean nothing about a real GPU.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "embeddinggemma2-webgpu-tail" / "run"))
from run_demo import CHROME, CHROME_ARGS, free_port, serve_from_cache, wait_for_server  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ort", action="store_true", help="also run onnxruntime-web (slow on SwiftShader)")
    a = ap.parse_args()
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                              cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/{ROOT.name}/web/index.html?auto=1&reps={a.reps}&ort={int(a.ort)}"
    errors = []
    try:
        wait_for_server(url)
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
            page = b.new_page()
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.route("https://cdn.jsdelivr.net/**", serve_from_cache)
            page.goto(url)
            page.wait_for_function("['done', 'error'].includes(window.result.phase)", timeout=3_600_000)
            res = page.evaluate("window.result")
            b.close()
    finally:
        server.terminate()
    res["console_errors"] = errors
    (ROOT / "results" / "check_swiftshader.json").write_text(json.dumps(res, indent=1))
    if res["phase"] != "done":
        print(res.get("error"), errors)
        return 1
    w = res["wgsl"]
    ok = w["cos_single_min"] >= 0.9999 and w["cos_batch_min"] >= 0.9999
    print("cos single", w["cos_single"], "\nbatch min", w["cos_batch_min"], "\nsummary", res["summary"],
          "\nerrors", errors, "\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
