"""Torch-free encoder for hotchpotch/bekko-embedding-v1-{a8m,a25m}.

ONNX Runtime + tokenizers only. The model is mean-pooled and L2-normalized by
hand; bekko applies **no query/document prefixes**, unlike Jina v5 nano.

Matryoshka: truncate then re-normalize (384 -> 256/128/64).

Mini-batched by design: METHODS.md records a one-shot `encode(all_docs)` OOMing
at ~26 GB on the attention-mask Expand broadcast.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODELS_ROOT = Path(os.environ.get("BEKKO_BENCH_MODELS", "/home/user/models"))


class BekkoEncoder:
    """Mean-pooled, L2-normalized bekko encoder over an ONNX graph."""

    full_dim = 384
    prompts = {"query": "", "document": ""}  # bekko uses no prefixes

    def __init__(
        self,
        variant: str = "a8m",
        onnx_name: str = "onnx/model.onnx",
        max_len: int = 512,
        threads: int | None = None,
    ) -> None:
        self.variant = variant
        self.root = MODELS_ROOT / f"bekko-{variant}"
        self.onnx_path = self.root / onnx_name
        self.max_len = max_len

        self.tok = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=self._pad_id(), pad_token="<pad>")

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(self.onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def _pad_id(self) -> int:
        import json

        cfg = json.loads((self.root / "config.json").read_text())
        return int(cfg.get("pad_token_id", 0))

    def model_bytes(self) -> int:
        return self.onnx_path.stat().st_size

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 8,
        dim: int | None = None,
        progress: bool = False,
        sort_by_length: bool = True,
    ) -> np.ndarray:
        """Return (N, dim) float32, L2-normalized. `dim` truncates (Matryoshka).

        ``sort_by_length`` groups similar-length inputs into a batch before
        encoding and restores the caller's order afterwards. Padding is to the
        batch's longest sequence, and this corpus runs median 322 / p90 927
        tokens, so corpus-order batching wastes 1.47x in padded tokens; sorting
        measured **1.25x** end-to-end with bit-identical output.
        """
        if sort_by_length and len(texts) > batch_size:
            order = np.argsort([len(t) for t in texts], kind="stable")
            packed = self.encode(
                [texts[i] for i in order], batch_size=batch_size,
                dim=dim, progress=progress, sort_by_length=False,
            )
            out = np.empty_like(packed)
            out[order] = packed
            return out
        out = np.empty((len(texts), self.full_dim), dtype=np.float32)
        t0 = time.time()
        for s in range(0, len(texts), batch_size):
            batch = texts[s : s + batch_size]
            encs = self.tok.encode_batch(batch)
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            feed = {k: v for k, v in feed.items() if k in self.input_names}
            hidden = self.sess.run(None, feed)[0]  # (B, T, H)
            m = mask.astype(np.float32)[..., None]
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out[s : s + len(batch)] = pooled
            if progress and (s // batch_size) % 20 == 0:
                print(
                    f"  {self.variant}: {s + len(batch)}/{len(texts)} "
                    f"({time.time() - t0:.1f}s)",
                    flush=True,
                )
        self.last_wall = time.time() - t0
        return matryoshka(out, dim)


def matryoshka(vecs: np.ndarray, dim: int | None = None) -> np.ndarray:
    """Truncate to `dim` coordinates then re-normalize to unit length."""
    v = vecs[:, :dim] if dim else vecs
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.clip(n, 1e-9, None)).astype(np.float32)


def count_tokens(enc: BekkoEncoder, texts: list[str]) -> int:
    return sum(len(e.ids) for e in enc.tok.encode_batch(texts))
