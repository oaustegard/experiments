"""Score ONNX variants built by run/export.py on the Bluesky task (CPU onnxruntime, 4 threads).

Usage: .venv/bin/python -I run/eval_onnx.py NAME [variant ...]   (default: every variant dir)

Per variant: model bytes (raw and gzip -9, ONNX + tokenizer.json), hit@10 / hit@1 on its own
remex 2-bit codes, paired difference to EG2's committed codes and to this model's fp32 ONNX,
min/median cosine of query vectors to the fp32 ONNX, and single-query CPU latency.
Writes results/onnx_NAME.json (merged per variant).
"""
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen import POSTS, Q, Index, boot_ci, native, overlap, perquery  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class Enc:
    def __init__(self, d, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(native.MAX_TOKENS)
        meta = json.loads((d / "meta.json").read_text())
        p = meta.get("prompts") or {}
        self.qp = meta.get("query_prompt", p.get("query", ""))
        self.dp = meta.get("doc_prompt", p.get("document", p.get("passage", "")))

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
            e = self.sess.run(None, {"input_ids": ids, "attention_mask": attn})[0]
            if out is None:
                out = np.empty((len(texts), e.shape[1]), np.float32)
            out[rows] = e / np.linalg.norm(e, axis=1, keepdims=True)
        return out


def nbytes(d):
    """Raw and brotli-11 bytes (what a static host serves), ONNX + tokenizer.json."""
    import brotli
    m, t = (d / "model.onnx").read_bytes(), (d / "tokenizer.json").read_bytes()
    return len(m) + len(t), len(brotli.compress(m, quality=11)) + len(brotli.compress(t, quality=11))


def main():
    name, variants = sys.argv[1], sys.argv[2:]
    base = ROOT / "out" / "models" / name
    variants = variants or ["fp32"] + sorted(p.name for p in base.iterdir() if (p / "model.onnx").exists() and p.name != "fp32")
    posts = [json.loads(l) for l in POSTS.open()]
    texts = [p["text"] or "" for p in posts]
    synth = [json.loads(l) for l in (Q / "synth_test.jsonl").open()]
    real = [json.loads(l) for l in (Q / "real_test.jsonl").open()]
    src = np.array([r["post"] for r in synth])
    ix = Index()
    eg2_s, eg2_r = ix.topk(np.load(Q / "synth_test.eg2.npy")), ix.topk(np.load(Q / "real_test.eg2.npy"))
    hit_eg2 = (eg2_s == src[:, None]).any(1).astype(float)
    path = ROOT / "results" / f"onnx_{name}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    ref = {}
    for v in variants:
        d = base / v
        enc = Enc(d)
        t0 = time.time()
        X = enc.embed(texts, enc.dp)
        tdoc = time.time() - t0
        Qs, Qr = enc.embed([r["q"] for r in synth], enc.qp), enc.embed([r["q"] for r in real], enc.qp)
        (ts, b), (tr, _) = native.rx2_topk(X, Qs), native.rx2_topk(X, Qr)
        h = (ts == src[:, None]).any(1).astype(float)
        perquery(f"onnx_{name}_{v}", h)
        raw, gz = nbytes(d)
        lat = []
        for q in [r["q"] for r in real][:200]:
            t1 = time.perf_counter()
            enc.embed([q], enc.qp)
            lat.append(time.perf_counter() - t1)
        r = {"mb": round(raw / 1e6, 2), "br_mb": round(gz / 1e6, 2), "dim": X.shape[1], "bytes_per_doc": b,
             "vocab": enc.tok.get_vocab_size(), "doc_encode_s": round(tdoc), "query_ms_median": round(float(np.median(lat)) * 1000, 2),
             "hit@10": boot_ci(h), "minus_eg2": native.paired(h, hit_eg2), "hit@1": float((ts[:, 0] == src).mean()),
             "real_overlap_with_eg2": boot_ci(overlap(tr, eg2_r))}
        if v == "fp32":
            ref = {"h": h, "Qs": Qs, "Qr": Qr, "tr": tr}
            scr = ROOT / "out" / "screen" / json.loads((d / "meta.json").read_text())["hf"].replace("/", "__")
            if (scr / "synth.npy").exists():  # parity with the torch screen vectors
                c = (np.load(scr / "synth.npy") * Qs).sum(1)
                r["cos_to_torch_synth"] = [float(c.min()), float(np.median(c))]
        elif ref:
            for tag, c in (("synth", (ref["Qs"] * Qs).sum(1)), ("real", (ref["Qr"] * Qr).sum(1))):
                r[f"cos_to_fp32_{tag}"] = {"min": float(c.min()), "p01": float(np.percentile(c, 1)),
                                           "p05": float(np.percentile(c, 5)), "median": float(np.median(c))}
            r["minus_fp32"] = native.paired(h, ref["h"])
            r["real_overlap_with_fp32"] = boot_ci(overlap(tr, ref["tr"]))
        res[v] = r
        print(name, v, json.dumps(r), flush=True)
        path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
