"""Native CPU baseline for the browser bench: remex 2.1.0 (numpy) and onnxruntime on this machine.

Usage: python3 -I run/native_bench.py [--reps 3]

Same sizes as web/bench_tail.html and the demo bench:
  - embed: model_q4.onnx on onnxruntime's CPU provider, the first 32 demo documents as one batch
    (all cores, and pinned to one thread)
  - encode: remex Quantizer(768, bits=2, seed=42, mean) on the 100k real EG2 Quora vectors
  - scan: 20 queries over the 100k codes, top-10, with remex search_adc and search_batch
  - float baseline: the same 20 queries as an fp32 matrix product over the raw vectors, top-10
Needs out/eg2q (scripts/embstore.py get quora/embeddinggemma-2/fp32-768-first100k) and out/hf
(run/ref_real.py). Writes results/native_bench.json.
"""
import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import remex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "out" / "eg2q"
HF = ROOT / "out" / "hf"
NQ, K, BATCH = 20, 10, 32


def timed(fn, reps):
    fn()                                   # warm-up
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) * 1000)
    return {"median": round(float(np.median(runs)), 2), "runs": [round(r, 2) for r in runs]}


def demo_docs(n):
    js = "import('./web/corpus.js').then(m => console.log(JSON.stringify(m.CORPUS.slice(0, %d))))" % n
    out = subprocess.run(["node", "-e", js], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return ["title: none | text: " + t for t in json.loads(out)]


def embed_bench(reps):
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    texts = demo_docs(BATCH)
    enc = [tok.encode(t).ids for t in texts]
    S = max(map(len, enc))
    ids = np.zeros((BATCH, S), np.int64)
    attn = np.zeros((BATCH, S), np.int64)
    for b, row in enumerate(enc):
        ids[b, : len(row)] = row
        attn[b, : len(row)] = 1
    empty = np.zeros((0, 512), np.float32)
    feeds = {"input_ids": ids, "attention_mask": attn,
             "image_features": empty, "video_features": empty, "audio_features": empty}
    out = {"batch": BATCH, "seq_len": S}
    for label, threads in (("all_cores", 0), ("one_thread", 1)):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(HF / "model_q4.onnx"), so, providers=["CPUExecutionProvider"])
        out[f"embed_ms_{label}"] = timed(lambda: sess.run(["sentence_embedding"], feeds), reps)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    X = np.concatenate([np.load(f) for f in sorted(glob.glob(str(VEC / "shards" / "docs_*.npy")))]).astype(np.float32)
    Qs = np.load(VEC / "q_dense.npy").astype(np.float32)[:NQ]
    mu = X.mean(0).astype(np.float32)
    qz = remex.Quantizer(768, bits=2, seed=42, mean=mu)

    res = {
        "machine": {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
                    "numpy": np.__version__, "remex": remex.__version__ if hasattr(remex, "__version__") else None,
                    "onnxruntime": ort.__version__},
        "n": len(X), "nq": NQ, "k": K,
    }
    res["encode_ms"] = timed(lambda: qz.encode(X), a.reps)
    cv = qz.encode(X)
    res["scan_search_adc_ms"] = timed(lambda: [qz.search_adc(cv, q, k=K) for q in Qs], a.reps)
    res["scan_search_batch_ms"] = timed(lambda: qz.search_batch(cv, Qs, k=K), a.reps)

    def fp32():
        s = Qs @ X.T
        top = np.argpartition(-s, K, axis=1)[:, :K]
        return np.take_along_axis(s, top, 1)
    res["scan_fp32_matmul_ms"] = timed(fp32, a.reps)
    res.update(embed_bench(a.reps))

    out = ROOT / "results" / "native_bench.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
