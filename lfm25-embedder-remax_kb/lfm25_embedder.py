"""LFM2.5-Embedding-350M embedder for remax_kb.

Conforms to the remax_kb ``Embedder`` protocol (see remax_kb/embedders.py).
Unlike JinaONNXEmbedder (847 MB download) or GeminiEmbedder (API key + network
round-trip per encode), this runs fully in-process on CPU via sentence-
transformers: no API key, no network at query time after the one-time model
pull, deterministic. ~350M params, 1024-dim CLS-pooled, query:/document:
prompt prefixes (handled by ST's registered prompts).
"""
from __future__ import annotations

from typing import Any

import numpy as np

LFM25_EMB_MODEL_ID = "LiquidAI/LFM2.5-Embedding-350M"
LFM25_EMB_FULL_DIM = 1024


def _patch_shortconv_kwargs() -> None:
    """The model's bidirectional remote code rebinds Lfm2ShortConv.slow_forward
    to a handler that predates transformers' `seq_idx` kwarg (variable-length
    packing). transformers>=5.x passes seq_idx through, so calls blow up with
    TypeError. Our batches are plain padded sequences — seq_idx is irrelevant —
    so wrap slow_forward to drop unexpected kwargs. Idempotent; run after the
    remote code installs its own patch (i.e. after model load)."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    cur = Lfm2ShortConv.slow_forward
    if getattr(cur, "_drops_extra_kwargs", False):
        return

    def slow_forward(self, hidden_states, past_key_values=None,
                     cache_position=None, attention_mask=None, **_ignored):
        return cur(self, hidden_states, past_key_values=past_key_values,
                   cache_position=cache_position, attention_mask=attention_mask)

    slow_forward._drops_extra_kwargs = True
    Lfm2ShortConv.slow_forward = slow_forward


class LFM25Embedder:
    model_id = LFM25_EMB_MODEL_ID
    model_revision = ""
    task_adapter = "retrieval"
    pooling = "cls"
    full_dim = LFM25_EMB_FULL_DIM
    normalize_l2 = True
    release_url = None          # local model, no asset URL in the .kb manifest
    release_sha256 = None
    # remax_kb passes prompt="query"|"document"; LFM2.5's ST config registers
    # prompt_names "query"/"document", so the keys map straight through.
    prompts = {"query": "query: ", "document": "document: "}

    def __init__(self, *, max_seq_length: int = 512):
        self._model = None
        self._max_seq_length = int(max_seq_length)

    def fingerprint(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_adapter": self.task_adapter,
            "pooling": self.pooling,
            "full_dim": self.full_dim,
        }

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(self.model_id, trust_remote_code=True)
        m.max_seq_length = self._max_seq_length
        _patch_shortconv_kwargs()
        self._model = m
        return m

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
        if prompt not in self.prompts:
            raise ValueError(
                f"unknown prompt {prompt!r}; expected one of {list(self.prompts)}"
            )
        if not texts:
            return np.zeros((0, self.full_dim), dtype=np.float32)
        model = self._load()
        vecs = model.encode(
            list(texts),
            prompt_name=prompt,          # ST applies the registered query:/document: prefix
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
