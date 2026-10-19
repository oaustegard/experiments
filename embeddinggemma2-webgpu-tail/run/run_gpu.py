"""Run web/test.html in Chromium (SwiftShader WebGPU) and write out/gpu/.

Usage: python3 -I run/run_gpu.py [--fixtures fixtures] [--out out/gpu] [--rot rht|dense]

Starts `python3 -m http.server` on a free port (cwd = project root), loads
/web/test.html, awaits window.runTail(), decodes the base64 outputs into
out/gpu/, and writes out/gpu/meta.json. The server is stopped by PID.
"""
import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
CHROME_ARGS = [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--use-angle=swiftshader",
    "--use-webgpu-adapter=swiftshader",
    "--enable-features=Vulkan",
]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout: float = 20.0) -> None:
    url = f"http://127.0.0.1:{port}/web/test.html"
    end = time.time() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            if time.time() > end:
                raise SystemExit(f"http.server did not come up on port {port}")
            time.sleep(0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="fixtures", help="fixture directory, relative to the project root")
    ap.add_argument("--out", default="out/gpu", help="output directory, relative to the project root")
    ap.add_argument("--rot", default="rht", choices=["rht", "dense"], help="rotation the kernels apply")
    args = ap.parse_args()
    OUT = ROOT / args.out
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    page_errors: list[str] = []
    console_errors: list[str] = []
    try:
        wait_for_server(port)
        url = f"http://127.0.0.1:{port}/web/test.html?fx={args.fixtures}&rot={args.rot}"
        t0 = time.time()
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
            try:
                page = browser.new_page()
                page.set_default_timeout(0)
                page.on(
                    "console",
                    lambda m: console_errors.append(f"[{m.type}] {m.text}")
                    if m.type in ("error", "warning") else None,
                )
                page.on("pageerror", lambda e: page_errors.append(str(e)))
                page.goto(url, wait_until="load", timeout=60_000)
                page.wait_for_function("window.tailReady === true", timeout=60_000)
                result = page.evaluate("() => window.runTail()")
            finally:
                browser.close()
        wall_s = round(time.time() - t0, 2)
    finally:
        server.terminate()
        server.wait(timeout=10)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, b64 in result["files"].items():
        (OUT / name).write_bytes(base64.b64decode(b64))

    meta = {
        "url": url,
        "wall_clock_s": wall_s,
        "n": result["n"],
        "nq": result["nq"],
        "k": result["k"],
        "capacity": result["capacity"],
        "timings_ms": result["timings_ms"],
        "adapter": result["adapter"],
        "colsum_mu_max_abs_err": result["colsum_mu_max_abs_err"],
        "topk_gpu_host_mismatch": result["topk_gpu_host_mismatch"],
        "rotation": result["rotation"],
        "uncaptured_errors": result["uncaptured_errors"],
        "page_errors": page_errors,
        "console_errors": console_errors,
        "note": "SwiftShader (software) WebGPU: timings are not real-GPU speed.",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")

    print(f"wrote {len(result['files'])} files + meta.json to {OUT}")
    print(json.dumps({"timings_ms": result["timings_ms"], "adapter": result["adapter"],
                      "uncaptured_errors": result["uncaptured_errors"]}, indent=1))
    for line in page_errors + console_errors:
        print("PAGE:", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
