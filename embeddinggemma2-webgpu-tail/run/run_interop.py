"""Run interop/test_interop.html in Chromium (SwiftShader WebGPU) and write out/interop/.

Usage: python3 -I run/run_interop.py

Starts `python3 -m http.server` on a free port (cwd = project root), loads
/interop/test_interop.html, awaits window.runInterop(), decodes the base64
outputs into out/interop/, and writes out/interop/meta.json. The server is
stopped by PID. Exits nonzero if any embedding was read back to the CPU
during the encode path, or if the page reported errors.

Check afterwards with: python3 -I oracle/check_tail.py fixtures out/interop
"""
import base64
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "interop"
PAGE = "/interop/test_interop.html"
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
        url = f"http://127.0.0.1:{port}{PAGE}"
        wait_for_server(url)
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
                page.wait_for_function("window.interopReady === true", timeout=60_000)
                result = page.evaluate("() => window.runInterop()")
            finally:
                browser.close()
        wall_s = round(time.time() - t0, 2)
    finally:
        server.terminate()
        server.wait(timeout=10)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, b64 in result["files"].items():
        (OUT / name).write_bytes(base64.b64decode(b64))

    gd = result["getdata_calls"]
    no_readback = gd["before_encode"] == 0 and gd["during_encode"] == 0
    meta = {
        "url": url,
        "wall_clock_s": wall_s,
        "n": result["n"],
        "nq": result["nq"],
        "k": result["k"],
        "capacity": result["capacity"],
        "timings_ms": result["timings_ms"],
        "adapter": result["adapter"],
        "session_inputs": result["session_inputs"],
        "session_outputs": result["session_outputs"],
        "getdata_calls": gd,
        "no_embedding_readback_on_encode_path": no_readback,
        "uncaptured_errors": result["uncaptured_errors"],
        "page_errors": page_errors,
        "console_errors": console_errors,
        "note": "SwiftShader (software) WebGPU: timings are not real-GPU speed.",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")

    print(f"wrote {len(result['files'])} files + meta.json to {OUT}")
    print(json.dumps({"timings_ms": result["timings_ms"], "adapter": result["adapter"],
                      "session_inputs": result["session_inputs"],
                      "session_outputs": result["session_outputs"],
                      "getdata_calls": gd,
                      "no_embedding_readback_on_encode_path": no_readback,
                      "uncaptured_errors": result["uncaptured_errors"]}, indent=1))
    for line in page_errors + console_errors:
        print("PAGE:", line)

    if not no_readback:
        print("FAIL: an embedding was read back to the CPU on the encode path")
        return 1
    if result["uncaptured_errors"] or page_errors:
        print("FAIL: page or uncaptured WebGPU errors")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
