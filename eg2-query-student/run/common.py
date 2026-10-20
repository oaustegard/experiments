"""Shared pieces: the published 2-bit Bluesky index, the EG2 teacher, the overlap metric.

The doc side is fixed: the 61,686 posts in ../embeddinggemma2-webgpu-tail/web/bsky, encoded
once by EG2 (model_q4.onnx) into remex 2-bit codes (Quantizer(768, bits=2, seed=42,
mean=mu)). Every arm here changes only how a query becomes a 768-d vector.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import remex
from remex.packing import unpack
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
HF = OUT / "hf"
IDX = ROOT.parent / "embeddinggemma2-webgpu-tail" / "web" / "bsky"
QUERY = "task: search result | query: "
K = 10


class Index:
    """The committed codes, scored the way remex search_adc scores them.

    search_adc ranks by (R q) . c[idx] * scale + bias + q . mu; the last term is constant
    per query, so it is dropped. Directions are decoded once (n x 768 float32, 190 MB) so a
    batch of queries is one matmul.
    """

    def __init__(self):
        meta = json.loads((IDX / "index.json").read_text())
        n, d = meta["n"], meta["d"]
        mu = np.array(json.loads(meta["mu"]) if isinstance(meta["mu"], str) else meta["mu"], np.float32)
        self.qz = remex.Quantizer(d, bits=2, seed=42, mean=mu)
        packed = np.frombuffer((IDX / "codes.u8.txt").read_bytes(), np.uint8).reshape(n, -1)
        idx = unpack(packed.reshape(-1), 2, n * d).reshape(n, d)
        aux = np.frombuffer((IDX / "aux.f32.txt").read_bytes(), "<f4").reshape(n, 2)
        self.scale, self.bias = aux[:, 0].copy(), aux[:, 1].copy()
        self.dirs = self.qz.centroids.astype(np.float32)[idx]
        self.n, self.d = n, d

    def scores(self, Q):
        Qr = self.qz._rotate_rows(np.asarray(Q, np.float32)).astype(np.float32)
        return (Qr @ self.dirs.T) * self.scale + self.bias

    def topk(self, Q, k=K, chunk=256):
        out = np.empty((len(Q), k), np.int64)
        for s in range(0, len(Q), chunk):
            sc = self.scores(Q[s:s + chunk])
            part = np.argpartition(-sc, k, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sc, part, 1), axis=1)
            out[s:s + chunk] = np.take_along_axis(part, order, 1)
        return out


class Teacher:
    """EG2 model_q4.onnx on CPU, the graph that built the doc codes."""

    def __init__(self, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(HF / "model_q4.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
        t = json.loads((HF / "tokenizer.json").read_text())
        self.specials = sorted((a["content"] for a in t["added_tokens"] if a.get("special")),
                               key=len, reverse=True)

    def clean(self, text):
        for sp in self.specials:
            if sp in text:
                text = text.replace(sp, " ")
        return text

    def embed(self, texts, prompt=QUERY, max_tokens=128, batch=64):
        texts = [prompt + self.clean(t) for t in texts]
        enc = [self.tok.encode(t).ids[:max_tokens] for t in texts]
        order = np.argsort([len(e) for e in enc])
        out = np.empty((len(texts), 768), np.float32)
        empty = np.zeros((0, 512), np.float32)
        for s in range(0, len(order), batch):
            rows = order[s:s + batch]
            S = max(len(enc[r]) for r in rows)
            ids = np.zeros((len(rows), S), np.int64)
            attn = np.zeros((len(rows), S), np.int64)
            for b, r in enumerate(rows):
                ids[b, :len(enc[r])] = enc[r]
                attn[b, :len(enc[r])] = 1
            e = self.sess.run(["sentence_embedding"], {
                "input_ids": ids, "attention_mask": attn, "image_features": empty,
                "video_features": empty, "audio_features": empty})[0]
            out[rows] = e / np.linalg.norm(e, axis=1, keepdims=True)
        return out


def overlap(a, b):
    """Per-query |top10(a) ∩ top10(b)| / 10."""
    return np.array([len(set(x) & set(y)) / K for x, y in zip(a, b)])


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)
