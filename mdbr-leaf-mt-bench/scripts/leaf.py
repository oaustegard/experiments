"""Torch-free encoder for MongoDB/mdbr-leaf-mt (ONNX Runtime + tokenizers).

The sentence-transformers pipeline is Transformer -> mean-pool -> Dense
(384 -> 1024, no bias, identity activation). The shipped ONNX graphs export
only the BertModel, so pooling and the Dense projection are applied here by
hand; the Dense weight is read from ``2_Dense/model.safetensors`` without a
torch dependency.

Prefixes per ``config_sentence_transformers.json``: queries get
"Represent this sentence for searching relevant passages: ", documents get
nothing. Matryoshka truncation is truncate-then-renormalize, same as every
other encoder in this bench family.
"""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODELS_ROOT = Path(os.environ.get("LEAF_BENCH_MODELS", "/home/user/models"))

PROMPTS = {
    "query": "Represent this sentence for searching relevant passages: ",
    "document": "",
}


def _load_dense(root: Path) -> np.ndarray:
    """Read the (1024, 384) linear weight from 2_Dense/model.safetensors."""
    raw = (root / "2_Dense" / "model.safetensors").read_bytes()
    hlen = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + hlen])
    (name, meta), = [(k, v) for k, v in header.items() if k != "__metadata__"]
    assert meta["dtype"] == "F32", meta
    s, e = meta["data_offsets"]
    w = np.frombuffer(raw[8 + hlen + s : 8 + hlen + e], dtype=np.float32)
    return w.reshape(meta["shape"])  # (out=1024, in=384)


class LeafMTEncoder:
    """Mean-pooled + Dense-projected + L2-normalized mdbr-leaf-mt encoder."""

    full_dim = 1024
    model_id = "MongoDB/mdbr-leaf-mt"

    def __init__(
        self,
        onnx_name: str = "onnx/model.onnx",
        max_len: int = 512,
        threads: int | None = None,
    ) -> None:
        self.root = MODELS_ROOT / "leaf-mt"
        self.onnx_path = self.root / onnx_name

        self.tok = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(self.onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.dense = _load_dense(self.root)  # (1024, 384)

    def model_bytes(self) -> int:
        data = self.onnx_path.with_suffix(self.onnx_path.suffix + "_data")
        n = self.onnx_path.stat().st_size
        if data.exists():
            n += data.stat().st_size
        n += (self.root / "2_Dense" / "model.safetensors").stat().st_size
        return n

    def encode(
        self,
        texts: list[str],
        *,
        prompt: str = "document",
        batch_size: int = 8,
        dim: int | None = None,
        progress: bool = False,
        sort_by_length: bool = True,
    ) -> np.ndarray:
        """Return (N, dim) float32, L2-normalized. `dim` truncates (Matryoshka)."""
        if sort_by_length and len(texts) > batch_size:
            order = np.argsort([len(t) for t in texts], kind="stable")
            packed = self.encode(
                [texts[i] for i in order], prompt=prompt, batch_size=batch_size,
                dim=dim, progress=progress, sort_by_length=False,
            )
            out = np.empty_like(packed)
            out[order] = packed
            return out
        pre = PROMPTS[prompt]
        out = np.empty((len(texts), self.full_dim), dtype=np.float32)
        t0 = time.time()
        for s in range(0, len(texts), batch_size):
            batch = [pre + t for t in texts[s : s + batch_size]]
            encs = self.tok.encode_batch(batch)
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            feed = {k: v for k, v in feed.items() if k in self.input_names}
            hidden = self.sess.run(None, feed)[0]  # (B, T, 384)
            m = mask.astype(np.float32)[..., None]
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out[s : s + len(batch)] = pooled @ self.dense.T
            if progress and (s // batch_size) % 20 == 0:
                print(f"  leaf-mt: {s + len(batch)}/{len(texts)} "
                      f"({time.time() - t0:.1f}s)", flush=True)
        self.last_wall = time.time() - t0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.clip(n, 1e-9, None)
        return matryoshka(out, dim)


def matryoshka(vecs: np.ndarray, dim: int | None = None) -> np.ndarray:
    v = vecs[:, :dim] if dim else vecs
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.clip(n, 1e-9, None)).astype(np.float32)
