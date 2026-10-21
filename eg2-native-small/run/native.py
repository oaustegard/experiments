"""Native small embedders on the eg2-query-student Bluesky task: their own doc index, their own queries.

Usage: .venv/bin/python -I run/native.py [arm ...]      (default: every arm in ARMS)

#102 and #103 asked whether a small model can embed QUERIES into EG2's space. This asks the
prior question: does a small model with its OWN index (docs and queries both) do the task as
well as EG2 does, so the browser ships ~23 MB instead of 174 MB (or 29 MB for #103's k=4 cut)?

Task and data are #102's: the 61,686 AI-list Bluesky posts (text from
../eg2-query-student/out/posts.jsonl, hydrate_posts.py), and the 1,498 held-out synthetic
queries, each written from one post, so hit@10 = that post in the top 10. Docs truncated to
128 tokens like EG2's index. Two doc representations per arm:
  fp32   exact cosine over float vectors
  rx2    remex 2-bit codes (Quantizer(d, bits=2, seed=42, mean=doc mean)), scored by
         search_adc's form -- EG2's deployed config, so rx2 is the matched comparison
EG2's row is recomputed from the committed codes and #102's EG2 query vectors (expect 0.919).
Also reported: top-10 agreement with EG2 on the 1,000 real MS MARCO queries (agreement, not
quality: there are no labels) and CPU latency on this Mac. Writes results/native.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import remex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
Q = OUT / "q"
POSTS = ROOT.parent / "eg2-query-student" / "out" / "posts.jsonl"
IDX = ROOT.parent / "embeddinggemma2-webgpu-tail" / "web" / "bsky"
sys.path.insert(0, str(ROOT.parent / "eg2-query-student" / "run"))
from common import Index, boot_ci, overlap  # noqa: E402

MAX_TOKENS = 128
LEAF_Q = "Represent this sentence for searching relevant passages: "
ARMS = {  # name: (dir, onnx file, query prompt, doc prompt, pooled output or None for mean pool)
    "leaf-ir": ("leaf-ir", "onnx/model_quantized.onnx", LEAF_Q, "", "sentence_embedding"),
    "leaf-mt": ("leaf-mt", "onnx/model_quantized.onnx", LEAF_Q, "", "sentence_embedding"),
    "minilm": ("minilm", "onnx/model_int8.onnx", "", "", None),
    # leaf-wasm (experiments/leaf-wasm): MongoDB's fp32 graph, and the same graph carrying
    # leaf-wasm's int8 weights dequantized (run/patch_onnx.py), which computes what the JS
    # engine computes (min cos 0.99999993). model_mb for these reports the fp32 graph on disk,
    # not the 23.76 MB leaf-ir-i8.bin the browser downloads.
    "leaf-ir-fp32": ("../../leaf-wasm/out/hf", "onnx/model.onnx", LEAF_Q, "", "sentence_embedding"),
    "leaf-ir-wasm-i8": ("../../leaf-wasm/out/onnx-i8deq", "onnx/model.onnx", LEAF_Q, "", "sentence_embedding"),
}


def paired(x, y, n=2000, seed=0):
    d = x - y
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


class Encoder:
    def __init__(self, arm, threads=4):
        d, f, self.qp, self.dp, self.pooled = ARMS[arm]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(OUT / d / f), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(str(OUT / d / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(MAX_TOKENS)
        self.bytes = sum(p.stat().st_size for p in (OUT / d / "onnx").iterdir()) + (OUT / d / "tokenizer.json").stat().st_size

    def embed(self, texts, prompt, batch=64):
        enc = [e.ids for e in self.tok.encode_batch([prompt + t for t in texts])]
        order = np.argsort([len(e) for e in enc])
        out = None
        for s in range(0, len(order), batch):
            rows = order[s:s + batch]
            S = max(len(enc[r]) for r in rows)
            ids = np.zeros((len(rows), S), np.int64)
            attn = np.zeros((len(rows), S), np.int64)
            for b, r in enumerate(rows):
                ids[b, :len(enc[r])] = enc[r]
                attn[b, :len(enc[r])] = 1
            feed = {"input_ids": ids, "attention_mask": attn}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            if self.pooled:
                e = self.sess.run([self.pooled], feed)[0]
            else:
                h = self.sess.run(["last_hidden_state"], feed)[0]
                m = attn[..., None].astype(np.float32)
                e = (h * m).sum(1) / m.sum(1)
            if out is None:
                out = np.empty((len(texts), e.shape[1]), np.float32)
            out[rows] = e / np.linalg.norm(e, axis=1, keepdims=True)
        return out


def topk(scores, k=10):
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def rx2_topk(X, Qv, k=10):
    """Same scoring as common.Index, on this arm's own 2-bit codes."""
    qz = remex.Quantizer(X.shape[1], bits=2, seed=42, mean=X.mean(0).astype(np.float32))
    cv = qz.encode(X)
    scale, bias = qz._score_norms(cv).astype(np.float32), qz._row_bias(cv).astype(np.float32)
    dirs = qz.centroids.astype(np.float32)[cv.indices]
    out = []
    for s in range(0, len(Qv), 256):
        Qr = qz._rotate_rows(Qv[s:s + 256]).astype(np.float32)
        out.append(topk((Qr @ dirs.T) * scale + bias, k))
    return np.concatenate(out), X.shape[1] * 2 // 8 + 8


def latency(enc, queries, n=200):
    t = []
    for q in queries[:n]:
        t0 = time.perf_counter()
        enc.embed([q], enc.qp)
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)


def main():
    arms = sys.argv[1:] or list(ARMS)
    posts = [json.loads(l) for l in POSTS.open()]
    assert len(posts) == 61686 and all(p["i"] == i for i, p in enumerate(posts))
    texts = [p["text"] or "" for p in posts]
    synth = [json.loads(l) for l in (Q / "synth_test.jsonl").open()]
    real = [json.loads(l) for l in (Q / "real_test.jsonl").open()]
    src = np.array([r["post"] for r in synth])

    ix = Index()
    eg2_s = ix.topk(np.load(Q / "synth_test.eg2.npy"))
    eg2_r = ix.topk(np.load(Q / "real_test.eg2.npy"))
    hit_eg2 = (eg2_s == src[:, None]).any(1).astype(float)
    res_path = ROOT / "results" / "native.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res["EG2 q4 (rx2, 192 B/doc)"] = {"hit@10": boot_ci(hit_eg2),
                                       "missing_post_text": sum(p["text"] is None for p in posts)}
    print("EG2", res["EG2 q4 (rx2, 192 B/doc)"], flush=True)
    # Per-query hit arrays (synth_test order, 1 = source post in top 10) so other sessions can
    # pair against these arms. SYM metric: each arm on its own codes; EG2 is the shared anchor.
    per = {"metric": "SYM hit@10, own 2-bit codes (rx2) or fp32, 1,498 synth_test in file order",
           "ids": [r["id"] for r in synth], "EG2 q4 rx2": hit_eg2.astype(int).tolist()}
    (ROOT / "results" / "EG2_perquery.json").write_text(json.dumps(per))

    for arm in arms:
        enc = Encoder(arm)
        cache = OUT / arm / "docs.npy"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            X = np.load(cache)
        else:
            t0 = time.time()
            X = enc.embed(texts, enc.dp)
            np.save(cache, X)
            print(f"{arm}: docs {time.time() - t0:.0f} s", flush=True)
        Qs = enc.embed([r["q"] for r in synth], enc.qp)
        Qr = enc.embed([r["q"] for r in real], enc.qp)
        r = {"model_mb": round(enc.bytes / 1e6, 1), "dim": X.shape[1],
             "query_ms_median_4threads": latency(enc, [x["q"] for x in real])}
        for rep in ("fp32", "rx2"):
            if rep == "fp32":
                ts, tr, b = topk(Qs @ X.T), topk(Qr @ X.T), X.shape[1] * 4
            else:
                (ts, b), (tr, _) = rx2_topk(X, Qs), rx2_topk(X, Qr)
            h = (ts == src[:, None]).any(1).astype(float)
            r[rep] = {"bytes_per_doc": b, "hit@10": boot_ci(h), "hit@10_minus_eg2": paired(h, hit_eg2),
                      "hit@1": float((ts[:, 0] == src).mean()),
                      "real_overlap_with_eg2": boot_ci(overlap(tr, eg2_r))}
            per[f"{arm} {rep}"] = h.astype(int).tolist()
        (ROOT / "results" / f"{arm}_perquery.json").write_text(json.dumps(
            {"metric": per["metric"], "ids": per["ids"], "fp32": per[f"{arm} fp32"], "rx2": per[f"{arm} rx2"]}))
        res[arm] = r
        print(arm, json.dumps(r), flush=True)
        res_path.write_text(json.dumps(res, indent=1))
    res_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
