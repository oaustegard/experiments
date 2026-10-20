"""Does a shipped q4 student run in onnxruntime-web, and give the CPU vectors?

Usage: python3 -I run/smoke_web.py [--tag l2_k4_ep2_v32k]

Serves the repo root, routes the CDN through Python (headless Chromium has no route to the
egress proxy), loads out/export/<tag>/model_q4.onnx in web/smoke.html on the wasm and the
WebGPU (SwiftShader) execution providers, embeds 32 synth_test queries tokenized here with
the build's tokenizer.json, and compares each vector with onnxruntime CPU on the same file.
Writes results/smoke_web_<tag>.json. SwiftShader timings mean nothing; this checks the ops.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import Bench  # noqa: E402
from export import ort_embed  # noqa: E402
from model import ROOT, tokenize  # noqa: E402
from common import Teacher  # noqa: E402

REPO = ROOT.parent
sys.path.insert(0, str(REPO / "embeddinggemma2-webgpu-tail" / "run"))
from run_demo import CHROME, CHROME_ARGS, free_port, serve_from_cache, wait_for_server  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l2_k4_ep2_v32k")
    a = ap.parse_args()
    d = ROOT / "out" / "export" / a.tag
    T, B = Teacher(), Bench()
    enc = tokenize(T, B.texts("synth_test")[:32], remap=Tokenizer.from_file(str(d / "tokenizer.json")))
    (d / "smoke_ids.json").write_text(json.dumps(enc))
    ref = ort_embed(d / "model_q4.onnx", enc)
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                              cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = {"tag": a.tag, "n": len(enc)}
    rel = f"/eg2-layer-student/out/export/{a.tag}/"
    try:
        wait_for_server(f"http://127.0.0.1:{port}/eg2-layer-student/web/smoke.html")
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
            for ep in ("wasm", "webgpu"):
                page = b.new_page()
                page.route("https://cdn.jsdelivr.net/**", serve_from_cache)
                page.goto(f"http://127.0.0.1:{port}/eg2-layer-student/web/smoke.html"
                          f"?model={rel}&ids={rel}smoke_ids.json&ep={ep}")
                page.wait_for_function("window.result.phase !== 'loading'", timeout=900_000)
                st = page.evaluate("window.result")
                if st["phase"] == "done":
                    V = np.array(st["vectors"], np.float32)
                    cos = (V * ref).sum(1) / np.linalg.norm(V, axis=1)
                    res[ep] = {"phase": "done", "cos_to_cpu_min": float(cos.min()), "cos_to_cpu_mean": float(cos.mean())}
                else:
                    res[ep] = {"phase": st["phase"], "error": st.get("error", "")[:2000]}
                print(ep, json.dumps(res[ep])[:600], flush=True)
                page.close()
            b.close()
    finally:
        server.terminate()
    (ROOT / "results" / f"smoke_web_{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
