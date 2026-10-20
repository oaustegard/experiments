"""Correctness smoke test of web/index.html in headless Chromium (SwiftShader WebGPU).

Usage: python3 -I run/smoke_web.py

Serves the repo root, points the page at a local copy of the EG2 files (&hf=), routes the
CDN and Bluesky through Python (headless Chromium has no route to the egress proxy), and
runs a 60-item items file: 30 synth_test queries and 30 non-test post ids. Then reads the
saved shard with read_shards.read and compares every vector with the CPU teacher on the same
text: queries, and each post's snippet as built from out/posts.jsonl. SwiftShader timings
mean nothing; this checks the bytes, not the speed.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "embeddinggemma2-webgpu-tail" / "run"))
from build_web import snippet  # noqa: E402
from common import Teacher  # noqa: E402
from read_shards import read  # noqa: E402
from run_demo import CHROME, CHROME_ARGS, free_port, serve_from_cache, wait_for_server  # noqa: E402
from run_search import bsky_passthrough  # noqa: E402

OUT = ROOT / "out" / "smoke"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mirror = ROOT / "out" / "hfmirror"
    (mirror / "onnx").mkdir(parents=True, exist_ok=True)
    for name, sub in [("model_q4.onnx", "onnx"), ("model_q4.onnx_data", "onnx"),
                      ("tokenizer.json", ""), ("tokenizer_config.json", "")]:
        dst = mirror / sub / name
        if not dst.exists():
            os.symlink(ROOT / "out" / "hf" / name, dst)
    qs = [json.loads(l)["q"] for l in (ROOT / "out" / "queries" / "synth_test.jsonl").open()][:30]
    test_posts = {json.loads(l)["post"] for l in (ROOT / "out" / "queries" / "synth_test.jsonl").open()}
    texts = {json.loads(l)["i"]: json.loads(l)["text"] for l in (ROOT / "out" / "posts.jsonl").open()}
    pids = [i for i in range(0, 61686, 997) if i not in test_posts and texts[i] and len(texts[i]) >= 40][:30]
    (ROOT / "web" / "smoke_items.json").write_text(json.dumps({"shard": 30000, "items": qs + pids}))
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                              cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server(port)
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=CHROME, args=CHROME_ARGS)
            ctx = b.new_context(accept_downloads=True)
            page = ctx.new_page()
            page.route("https://cdn.jsdelivr.net/**", serve_from_cache)
            page.route("https://public.api.bsky.app/**", bsky_passthrough)
            page.goto(f"http://127.0.0.1:{port}/eg2-query-student/web/index.html"
                      f"?hf=/eg2-query-student/out/hfmirror/&items=smoke_items.json&parity_n=32")
            page.wait_for_function("window.egState && window.egState.phase !== 'loading'", timeout=600_000)
            with page.expect_download(timeout=1_800_000) as dl:
                page.click("#run")
            dl.value.save_as(OUT / "eg2_targets_00.bin")
            page.wait_for_function("['done','error','parity_failed'].includes(window.egState.phase)",
                                   timeout=600_000)
            st = page.evaluate("window.egState")
            b.close()
    finally:
        server.terminate()
    print(json.dumps({k: st[k] for k in ("phase", "error", "parity", "shards")}))
    h, V, status, lens = read(OUT / "eg2_targets_00.bin")
    T = Teacher()
    snips = [snippet(texts[i]) for i in pids]
    ref = T.embed(qs + snips)
    cos = (V * ref).sum(1)
    ok_len = [int(lens[30 + j]) == len(s) for j, s in enumerate(snips)]
    print(json.dumps({"status_all_1": bool(status.all()), "snippet_len_match": sum(ok_len),
                      "cos_queries_min": float(cos[:30].min()), "cos_posts_min": float(cos[30:].min()),
                      "cos_mean": float(cos.mean())}))


if __name__ == "__main__":
    main()
