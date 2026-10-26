"""Run web/wgsl.html in headless Chromium (SwiftShader WebGPU) and check the WGSL engine's vectors.

Usage: python3 -I run/check_wgsl.py [--pkgs e17c-t4-w8L63,e17c-t3-w8L63,leafirc-t4-w8L63] [--ort]

Needs each package's model.onnx in web/models/<pkg>/ (gitignored; unzip it from the
ettin-native-20261021 release, ARTIFACTS.md). Serves the experiments repo, answers jsDelivr from a
disk cache, opens wgsl.html and waits for window.check.done. Passes when every package's engine
vectors are at cosine >= 0.9999 to the fixtures (onnxruntime CPU on the same token ids).
SwiftShader timings mean nothing about a real GPU.
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
    ap.add_argument("--pkgs", default="e17c-t4-w8L63,e17c-t3-w8L63,leafirc-t4-w8L63")
    ap.add_argument("--ort", action="store_true", help="also run the onnxruntime-web fallback")
    a = ap.parse_args()
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                              cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    backends = "webgpu,ort" if a.ort else "webgpu"
    url = f"http://127.0.0.1:{port}/{ROOT.name}/web/wgsl.html?pkgs={a.pkgs}&backends={backends}"
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
            page.wait_for_function("window.check && window.check.done", timeout=3_600_000)
            res = page.evaluate("window.check")
            b.close()
    finally:
        server.terminate()
    res["console_errors"] = errors
    (ROOT / "results" / "check_wgsl_swiftshader.json").write_text(json.dumps(res, indent=1))
    ok = bool(res["rows"])
    for r in res["rows"]:
        # an engine failure silently falls back to onnxruntime-web, so engineError fails the check too
        good = "error" not in r and not r.get("engineError") and r.get("cosMin", 0) >= 0.9999
        ok &= good
        print(r["pkg"], r.get("backend"), "cos min", r.get("cosMin"), r.get("error", ""), "ok" if good else "FAIL")
    print("errors", errors, "\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
