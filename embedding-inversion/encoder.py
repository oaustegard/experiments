"""Torch-free bekko-embedding-v1-a8m encoder (ONNX Runtime + tokenizers), plus
the 1-bit code that stands in for the remax_kb index.

Lifted from `bekko-embedding-bench/scripts/bekko.py`, re-rooted on the
huggingface_hub cache so the experiment carries no /home/user literals. bekko
applies no query/document prefixes. Mean-pooled, L2-normalized.

`SignBits` is the centered-sign code: code = sign(e - mu), dequantized back to a
unit vector. remax_kb's stacked SimHash rotates first; the plain sign is the
same information budget (384 bits) and is what the capacity claim is about.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REPO = "hotchpotch/bekko-embedding-v1-a8m"
FILES = [
    "onnx/model.onnx", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "config.json",
]


def snapshot() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(REPO, allow_patterns=FILES))


class BekkoEncoder:
    full_dim = 384

    def __init__(self, max_len: int = 64, threads: int | None = None) -> None:
        root = snapshot()
        cfg = json.loads((root / "config.json").read_text())
        self.tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=int(cfg.get("pad_token_id", 0)), pad_token="<pad>")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(root / "onnx/model.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}
        outs = [o.name for o in self.sess.get_outputs()]
        self.out_name = next((o for o in outs if "sentence_embedding" in o), outs[0])
        self.pooled = "sentence_embedding" in self.out_name

    def n_tokens(self, texts: list[str]) -> list[int]:
        return [int(sum(e.attention_mask)) for e in self.tok.encode_batch(texts)]

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        """(N, 384) float32, L2-normalized. Length-sorted batching, order restored."""
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.zeros((len(texts), self.full_dim), dtype=np.float32)
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            encs = self.tok.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            y = self.sess.run([self.out_name], feed)[0]
            if not self.pooled:
                m = mask[..., None].astype(np.float32)
                y = (y * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            out[idx] = y
        return l2(out)


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


class SignBits:
    """1-bit centered-sign code. `fit` on training embeddings sets the center."""

    def __init__(self, mu: np.ndarray | None = None) -> None:
        self.mu = mu

    @classmethod
    def fit(cls, emb: np.ndarray) -> "SignBits":
        return cls(emb.mean(0).astype(np.float32))

    def code(self, emb: np.ndarray) -> np.ndarray:
        return (emb - self.mu >= 0)

    def dequant(self, emb: np.ndarray) -> np.ndarray:
        """Embedding -> code -> unit vector of +-1/sqrt(d). This is what the
        inverter and the verifier see under the bin1 condition."""
        return l2(np.where(self.code(emb), 1.0, -1.0).astype(np.float32))

    def save(self, path: Path) -> None:
        np.save(path, self.mu)

    @classmethod
    def load(cls, path: Path) -> "SignBits":
        return cls(np.load(path))


def condition(name: str, emb: np.ndarray, sb: SignBits | None) -> np.ndarray:
    """Map float embeddings to what the model sees under `name`."""
    if name == "float":
        return emb
    if name == "bin1":
        assert sb is not None
        return sb.dequant(emb)
    raise ValueError(name)
