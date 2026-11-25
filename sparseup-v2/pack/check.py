"""Run web/check.html in headless Chromium (SwiftShader WebGPU) and score the WGSL engine's vectors.

    python3 check.py --pack int8 [--only N] [--reps 3] [--no-timing]

Needs packs/<pack>/ (pack.py), web/fixtures/parity.json and cache/parity_ref.npz (fixture.py). Writes
results/wgsl_<pack>_swiftshader.json (raw rows, timing) and results/parity_<pack>.json (metrics).
SwiftShader runs WebGPU on the CPU: its timings say nothing about a real GPU.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "embeddinggemma2-webgpu-tail" / "run"))
from run_demo import (
    CHROME,
    CHROME_ARGS,
    free_port,
    serve_from_cache,
    wait_for_server,
)

if os.environ.get("CHROME_ARGS_JSON"):   # Kaggle: real-GPU flags (kaggle_parity.py)
    CHROME_ARGS = json.loads(os.environ["CHROME_ARGS_JSON"])


def metrics(a: np.ndarray, b: np.ndarray, k: int = 32) -> dict:
    """a, b: dense [V] vectors. Cosine, Jaccard of the top-k term sets, max |a - b|."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float(na == nb)
    ta = set(np.argsort(-a, kind="stable")[:k][a[np.argsort(-a, kind="stable")[:k]] > 0].tolist())
    tb = set(np.argsort(-b, kind="stable")[:k][b[np.argsort(-b, kind="stable")[:k]] > 0].tolist())
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 1.0
    return {"cos": cos, "jaccard32": jac, "max_abs": float(np.abs(a - b).max()),
            "nnz_a": int((a > 0).sum()), "nnz_b": int((b > 0).sum()), "support_jaccard": _support(a, b)}


def _support(a, b):
    sa, sb = set(np.flatnonzero(a)), set(np.flatnonzero(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def summarize(per: list[dict]) -> dict:
    out = {}
    for key in ("cos", "jaccard32", "max_abs", "support_jaccard"):
        x = np.array([p[key] for p in per])
        out[key] = {"min": float(x.min()), "median": float(np.median(x)), "mean": float(x.mean()), "max": float(x.max())}
    return out


def run_page(pack: str, start: int, only: int, reps: int, timing: bool) -> dict:
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                              cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = (f"http://127.0.0.1:{port}/{HERE.parent.name}/{HERE.name}/web/check.html?pack={pack}&reps={reps}"
           f"&start={start}&only={only}&timing={int(timing)}")
    errors = []
    try:
        wait_for_server(url)
        with sync_playwright() as p:
            exe = None if os.environ.get("CHROME_PATH") == "playwright" else CHROME
            b = p.chromium.launch(executable_path=exe, args=CHROME_ARGS, channel=os.environ.get("CHROME_CHANNEL"))
            page = b.new_page()
            page.on("console", lambda m: print("  page:", m.text, flush=True) if m.type in ("log", "error") else None)
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.route("https://cdn.jsdelivr.net/**", serve_from_cache)
            page.goto(url)
            page.wait_for_function("window.check && window.check.done", timeout=0, polling=2000)
            res = page.evaluate("window.check")
            b.close()
    finally:
        server.terminate()
    res["console_errors"] = errors
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="int8")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--only", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="score all saved chunks of this pack")
    ap.add_argument("--source", default="swiftshader", help="chunk set to score: swiftshader (this script) or e.g. t4 (run_node.mjs output in cache/wgsl_<pack>_t4/)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()
    sfx = "" if a.source == "swiftshader" else f"_{a.source}"
    chunks = HERE / "cache" / f"wgsl_{a.pack}{sfx}"
    chunks.mkdir(parents=True, exist_ok=True)
    if not a.merge and not sfx:
        res = run_page(a.pack, a.start, a.only, a.reps, not a.no_timing)
        (chunks / f"{a.start:03d}.json").write_text(json.dumps(res))
        if res.get("error"):
            print("ERROR", res["error"])
            return 1
    parts = [json.loads(f.read_text()) for f in sorted(chunks.glob("*.json"))]
    res = {"rows": [r for p in parts for r in p["rows"]], "console_errors": [e for p in parts for e in p["console_errors"]],
           "batch": [b for p in parts for b in p.get("batch", [])], "info": parts[0].get("info", {}),
           "timing": next((p["timing"] for p in parts if p.get("timing")), None)}
    order = {f["name"]: i for i, f in enumerate(json.loads((HERE / "web" / "fixtures" / "parity.json").read_text()))}
    res["rows"] = sorted({r["name"]: r for r in res["rows"]}.values(), key=lambda r: order[r["name"]])
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"wgsl_{a.pack}_{a.source}.json").write_text(json.dumps(res))
    fx = json.loads((HERE / "web" / "fixtures" / "parity.json").read_text())
    ref = np.load(HERE / "cache" / "parity_ref.npz")
    V = ref["fp32"].shape[1]
    wgsl = np.zeros((len(res["rows"]), V), dtype=np.float32)
    idx = [order[row["name"]] for row in res["rows"]]
    for r, row in enumerate(res["rows"]):
        wgsl[r, row["i"]] = row["w"]
        assert all(i < V for i in row["i"])   # padded ids never carry weight
    n = len(res["rows"])
    fx = [fx[i] for i in idx]
    arms = {"wgsl": wgsl, **{k: ref[k][idx] for k in ref.files}}
    pairs = [("wgsl", "fp32"), ("wgsl", a.pack), (a.pack, "fp32"), ("bf16", "fp32")]
    out = {"pack": a.pack, "n_texts": n, "kinds": {k: sum(f["kind"] == k for f in fx[:n]) for k in ("query", "document")},
           "tokens": {"min": min(f["n"] for f in fx[:n]), "max": max(f["n"] for f in fx[:n])},
           "js_tokenizer_matches": sum(r["tokOk"] for r in res["rows"]),
           "batch_vs_single_max_diff": max((b["maxDiff"] for b in res.get("batch", [])), default=None),
           "source": a.source, "timing": res.get("timing"), "adapter": res.get("info", {}).get("adapter"),
           "init_ms": res.get("info", {}).get("initMs"), "console_errors": res["console_errors"], "pairs": {}}
    for x, y in pairs:
        per = [metrics(arms[x][r], arms[y][r]) for r in range(n)]
        out["pairs"][f"{x}_vs_{y}"] = {"summary": summarize(per),
                                       "per_text": [{"name": fx[r]["name"], **per[r]} for r in range(n)]}
        s = out["pairs"][f"{x}_vs_{y}"]["summary"]
        print(f"{x:>5} vs {y:<5} cos min {s['cos']['min']:.6f} med {s['cos']['median']:.6f} | "
              f"jac32 min {s['jaccard32']['min']:.3f} mean {s['jaccard32']['mean']:.3f} | "
              f"max|d| max {s['max_abs']['max']:.4f} med {s['max_abs']['median']:.4f}")
    print("js tokenizer", out["js_tokenizer_matches"], "/", n, "| batch vs single", out["batch_vs_single_max_diff"],
          "| timing", json.dumps(out["timing"]))
    (HERE / "results" / f"parity_{a.pack}{sfx}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
