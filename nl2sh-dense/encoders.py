#!/usr/bin/env python3
"""Torch-free ONNX sentence encoders for the dense retrieval arm.

Two models, both chosen because a phone-sized shell helper has to carry the
encoder as well as the generator:

* **mdbr-leaf-mt int8** (`MongoDB/mdbr-leaf-mt`, `onnx/model_quantized.onnx`) —
  23M params, 1024-d, mean-pool + a 384->1024 Dense head. 24.9 MB on disk.
  `mdbr-leaf-mt-bench` measured this export as a statistical tie with bekko-a8m
  on retrieval at 5.2x smaller and 8.0 vs 10.8 ms per query.
* **bekko-embedding-v1-a8m** (`hotchpotch/bekko-embedding-v1-a8m`) — 384-d,
  mean-pool, no prefixes. This is the encoder `xr` already vendors, which is
  what issue #48 names; it is 164 MB with its tokenizer.

`LeafMTEncoder` follows `mdbr-leaf-mt-bench/scripts/leaf.py` — the shipped ONNX
graph exports the BertModel only, so pooling and the Dense projection are done
here and the Dense weight is read out of `2_Dense/model.safetensors` without a
torch dependency. `BekkoEncoder` follows `claude-workspace/scripts/xr.py`.

Model files live under `$NL2SH_DENSE_MODELS` (default `~/models`), one
directory per model, exactly as downloaded from the Hugging Face repo:

    <root>/leaf-mt/{model_quantized.onnx, model_quantized.onnx_data,
                    tokenizer.json, 2_Dense/model.safetensors}
    <root>/bekko-a8m/{model.onnx, tokenizer.json}
    <root>/minilm/{model.onnx, tokenizer.json}
"""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import numpy as np

MODELS_ROOT = Path(os.environ.get("NL2SH_DENSE_MODELS", Path.home() / "models"))


def _session(path: Path, threads: int | None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def _tokenizer(path: Path, pad_token: str, max_len: int = 512):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(pad_id=0, pad_token=pad_token)
    return tok


def _mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)[..., None]
    return (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.clip(n, 1e-9, None)).astype(np.float32)


class _OnnxEncoder:
    """Shared batching: length-sorted batches, restored to input order."""

    name = "?"
    dim = 0
    prompts = {"query": "", "document": ""}

    def encode(self, texts: list[str], *, prompt: str = "document",
               batch_size: int = 16, progress: bool = False) -> np.ndarray:
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        pre = self.prompts[prompt]
        t0 = time.time()
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            encs = self.tok.encode_batch([pre + texts[i] for i in idx])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            out[idx] = self._forward(feed, mask)
            if progress and (s // batch_size) % 100 == 0:
                done = min(s + batch_size, len(order))
                print(f"  {self.name}: {done}/{len(order)} ({time.time() - t0:.0f}s)",
                      flush=True)
        self.last_wall = time.time() - t0
        return _l2(out)


class LeafMTEncoder(_OnnxEncoder):
    name = "leaf-mt-int8"
    model_id = "MongoDB/mdbr-leaf-mt"
    dim = 1024
    # config_sentence_transformers.json: queries carry a prefix, documents none.
    prompts = {"query": "Represent this sentence for searching relevant passages: ",
               "document": ""}

    def __init__(self, onnx_name: str = "model_quantized.onnx",
                 threads: int | None = None) -> None:
        self.root = MODELS_ROOT / "leaf-mt"
        self.onnx_path = self.root / onnx_name
        self.tok = _tokenizer(self.root / "tokenizer.json", "[PAD]")
        self.sess = _session(self.onnx_path, threads)
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.dense = _load_dense(self.root)  # (1024, 384)

    def _forward(self, feed: dict, mask: np.ndarray) -> np.ndarray:
        hidden = self.sess.run(None, feed)[0]          # (B, T, 384)
        return _mean_pool(hidden, mask) @ self.dense.T

    def artifact_bytes(self) -> int:
        n = self.onnx_path.stat().st_size
        data = self.onnx_path.with_suffix(self.onnx_path.suffix + "_data")
        if data.exists():
            n += data.stat().st_size
        n += (self.root / "2_Dense" / "model.safetensors").stat().st_size
        n += (self.root / "tokenizer.json").stat().st_size
        return n


class BekkoEncoder(_OnnxEncoder):
    name = "bekko-a8m"
    model_id = "hotchpotch/bekko-embedding-v1-a8m"
    dim = 384
    prompts = {"query": "", "document": ""}

    def __init__(self, threads: int | None = None) -> None:
        self.root = MODELS_ROOT / "bekko-a8m"
        self.onnx_path = self.root / "model.onnx"
        self.tok = _tokenizer(self.root / "tokenizer.json", "<pad>")
        self.sess = _session(self.onnx_path, threads)
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def _forward(self, feed: dict, mask: np.ndarray) -> np.ndarray:
        return _mean_pool(self.sess.run(None, feed)[0], mask)

    def artifact_bytes(self) -> int:
        return (self.onnx_path.stat().st_size
                + (self.root / "tokenizer.json").stat().st_size)


class MiniLMEncoder(_OnnxEncoder):
    """all-MiniLM-L6-v2, int8 (`onnx/model_qint8_avx512_vnni.onnx`).

    The default small sentence encoder most projects reach for first, at 23.5 MB
    — here as the reference point, so "a dense arm helps" can be separated from
    "this particular dense model helps".
    """

    name = "minilm-l6-int8"
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    dim = 384
    prompts = {"query": "", "document": ""}

    def __init__(self, threads: int | None = None) -> None:
        self.root = MODELS_ROOT / "minilm"
        self.onnx_path = self.root / "model.onnx"
        self.tok = _tokenizer(self.root / "tokenizer.json", "[PAD]")
        self.sess = _session(self.onnx_path, threads)
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def _forward(self, feed: dict, mask: np.ndarray) -> np.ndarray:
        return _mean_pool(self.sess.run(None, feed)[0], mask)

    def artifact_bytes(self) -> int:
        return (self.onnx_path.stat().st_size
                + (self.root / "tokenizer.json").stat().st_size)


def _load_dense(root: Path) -> np.ndarray:
    """Read the (1024, 384) linear weight from 2_Dense/model.safetensors."""
    raw = (root / "2_Dense" / "model.safetensors").read_bytes()
    hlen = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + hlen])
    (name, meta), = [(k, v) for k, v in header.items() if k != "__metadata__"]
    assert meta["dtype"] == "F32", meta
    s, e = meta["data_offsets"]
    w = np.frombuffer(raw[8 + hlen + s : 8 + hlen + e], dtype=np.float32)
    return w.reshape(meta["shape"])


ENCODERS = {"leaf-mt-int8": LeafMTEncoder, "bekko-a8m": BekkoEncoder,
            "minilm-l6-int8": MiniLMEncoder}


def build(name: str, **kw) -> _OnnxEncoder:
    return ENCODERS[name](**kw)
