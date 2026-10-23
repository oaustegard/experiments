"""Open web/index.html in headless Chromium (SwiftShader WebGPU) and compare its tags with Python's.

Usage: python3 run/check.py   (after run/ref_tags.py .cache/img/*.jpg)

Serves this experiment's directory, answers jsDelivr and Hugging Face from a disk cache filled
through Python's HTTPS stack (headless Chromium has no route to the egress proxy), and calls
window.tagger.tagImage() on each reference image. Passes when every page embedding is at cosine
>= 0.999 to onnxruntime CPU and the top-5 tags agree. SwiftShader timings mean nothing.
"""
import hashlib
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "netcache"
CHROME = "/opt/pw-browsers/chromium"
ARGS = ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader",
        "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"]
CTYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json"}


def from_cache(route):
    url = route.request.url
    path = CACHE / hashlib.sha256(url.encode()).hexdigest()[:24]
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=600) as r:
            path.with_suffix(".part").write_bytes(r.read())
        path.with_suffix(".part").rename(path)
    default = "text/javascript" if "cdn.jsdelivr.net" in url else "application/octet-stream"  # bare package URL
    ctype = CTYPES.get(Path(url.split("?")[0]).suffix, default)
    route.fulfill(status=200, body=path.read_bytes(), headers={"content-type": ctype, "access-control-allow-origin": "*"})


def main():
    ref = json.loads((ROOT / "results/ref_tags.json").read_text())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    errors, out = [], {}
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=CHROME, args=ARGS)
            page = b.new_page()
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.route("https://cdn.jsdelivr.net/**", from_cache)
            page.route("https://huggingface.co/**", from_cache)
            page.goto(f"http://127.0.0.1:{port}/web/index.html?img=../.cache/img/000000000285.jpg")
            page.wait_for_function("window.tagger && window.tagger.ready", timeout=600_000)
            page.evaluate("window.tagger.ready")
            status = page.inner_text("#status")
            for name in ref:
                r = page.evaluate(f"window.tagger.tagImage('../.cache/img/{name}').then(r => ({{emb: r.emb, top: r.top}}))")
                out[name] = r
            b.close()
    finally:
        server.terminate()
    import numpy as np
    rows, ok = {}, True
    for name, r in out.items():
        cos = float(np.dot(r["emb"], ref[name]["emb"]) / np.linalg.norm(r["emb"]) / np.linalg.norm(ref[name]["emb"]))
        page_top = [t["tag"] for t in r["top"]]
        ref_top = [t for t, _ in ref[name]["top"][:5]]
        rows[name] = {"cos": round(cos, 6), "page_top5": page_top, "ref_top5": ref_top, "same_top5": page_top == ref_top}
        ok &= cos >= 0.999 and set(page_top) == set(ref_top)
        print(name, f"cos {cos:.6f}", page_top, "" if page_top == ref_top else f"ref {ref_top}")
    res = {"status": status, "images": rows, "console_errors": errors, "pass": bool(ok)}
    (ROOT / "results/check_swiftshader.json").write_text(json.dumps(res, indent=1))
    print(status, "\nerrors", errors, "\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
