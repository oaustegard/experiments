#!/usr/bin/env python3
"""bge-small-en-v1.5 ONNX embedder (CLS pooling + L2 norm), local weights only.

Env note: HF weight egress WORKS from the CCotw container (us.aws.cdn.hf.co
returns 206), contradicting memory 6b190772 which recorded us.gcp.cdn.hf.co as
blocked from the claude.ai container. Probed 2026-09-01.
"""
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "bge-small")
MAX_LEN = 512
# bge asks for this prefix on the QUERY side only; documents are encoded bare.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_sess = None
_tok = None


def _load():
    global _sess, _tok
    if _sess is None:
        _sess = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"),
                                     providers=["CPUExecutionProvider"])
        _tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        _tok.enable_truncation(max_length=MAX_LEN)
        _tok.enable_padding(length=None)
    return _sess, _tok


def encode(texts, batch_size=16, is_query=False):
    """Return (n, 384) float32 L2-normalised CLS embeddings."""
    sess, tok = _load()
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    names = {i.name for i in sess.get_inputs()}
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        encs = tok.encode_batch(chunk)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in names}
        last = sess.run(None, feed)[0]          # (b, seq, 384)
        cls = last[:, 0, :]                      # bge pools on CLS
        cls = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        out.append(cls.astype(np.float32))
    return np.vstack(out)


if __name__ == "__main__":
    q = encode(["totally unimodular matrix rounding error"], is_query=True)
    d = encode([
        "Linear Discrepancy of Totally Unimodular Matrices",
        "Integer and Unsplittable Multiflows in Series-Parallel Digraphs",
        "A survey of deep learning for image classification",
    ])
    print("dim", q.shape, d.shape)
    for s in (q @ d.T)[0]:
        print(f"  cos {s:.4f}")
