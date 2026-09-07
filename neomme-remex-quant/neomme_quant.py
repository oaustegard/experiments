#!/usr/bin/env python3
"""remex / remax vector quantization for NeoMME-Retriever embeddings, both heads.

NeoMME ships two heads from one forward pass:
  * dense_embeddings — (1024,) mean-pooled, L2-normalised, Matryoshka-trained at
    128/256/512/1024. Cosine scoring.
  * embeddings — (n_tokens, 128) per-token vectors, L2-normalised, MeanMaxSim
    scoring (each query token takes its best document token; mean over query tokens).

Neither head ships a quantized artifact upstream (checked 2026-09-07: safetensors
only, no onnx/int8/binary siblings, no derivative repos). This module gives both
heads a remex (multi-bit Lloyd-Max on a rotated sphere) and remax (1-bit
sign/SimHash, asymmetric float-query scoring) index, and reproduces
Sentence Transformers' own two size levers — Matryoshka truncation for the dense
head, HierarchicalTokenPooling for the token head — so the codecs are measured
against what the model already offers, not against nothing.

Numpy-only at query time. torch/scipy are needed only for token pooling
(re-uses sentence_transformers' `_hierarchical_pool_one` so pooled indexes are
byte-compatible with an ST pipeline).

Design notes
  * Rotate-then-quantize scrambles coordinate order, so Matryoshka truncation
    happens BEFORE remex/remax (upstream-prior-art rule: MRL first, quantize the
    residual second).
  * The multi-vector codec is per-token: a document of T tokens costs
    T * 128 * bits / 8 bytes (+ a float32 norm per token for remex, redundant on
    unit-norm tokens and only informative after pooling, where cluster means are
    not renormalised — ST leaves them that way, so we do too).
  * MaxSim over quantized tokens is computed in the codec's rotated space:
    decode the corpus once to float32 (cached), rotate each query's tokens with
    the same R, one GEMM per query, then a segmented max over document
    boundaries (np.maximum.reduceat) and a mean over query tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from remax import StackedSignBitQuantizer
from remex import Quantizer

MRL_DIMS = (128, 256, 512, 1024)


# --------------------------------------------------------------------------- helpers
def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def mrl_truncate(x: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation: keep the first `dim` coordinates, renormalise."""
    if dim not in MRL_DIMS:
        raise ValueError(f"dim must be one of {MRL_DIMS}, got {dim}")
    return l2n(np.asarray(x, dtype=np.float32)[..., :dim])


def signs_pm1(codes: np.ndarray, d_total: int) -> np.ndarray:
    """Bit-packed uint8 (n, d_total/8) -> float32 (n, d_total) in {-1, +1}."""
    bits = np.unpackbits(codes, axis=1)[:, :d_total]
    return (bits.astype(np.float32) * 2.0 - 1.0)


def seg_max_mean(S: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """MeanMaxSim from a (m_query_tokens, n_corpus_tokens) score matrix.

    offsets: (n_docs+1,) token boundaries. Returns (n_docs,) scores.
    """
    mx = np.maximum.reduceat(S, offsets[:-1], axis=1)   # (m, n_docs)
    return mx.mean(axis=0)


# --------------------------------------------------------------------------- dense head
@dataclass
class DenseIndex:
    """One dense-head index: fp32 / remex / remax over an MRL-truncated space.

    codec: "fp32" | "remex" | "remax"
    dim:   Matryoshka dimension (128/256/512/1024); truncation applied before coding.
    bits:  remex bits per coordinate (1-8); remax always 1 bit/dim.
    k:     remax stack depth (k independent sign codes; bytes = k*dim/8).
    query_mode (remax): "asym" = float query vs +/-1 codes (default),
                        "sym"  = query binarised too (Hamming-equivalent ranking).
    """
    codec: Literal["fp32", "remex", "remax"]
    dim: int = 1024
    bits: int = 4
    k: int = 1
    query_mode: Literal["asym", "sym"] = "asym"
    seed: int = 0
    rotation: str = "rht"

    def __post_init__(self):
        self.q: Optional[Quantizer | StackedSignBitQuantizer] = None
        if self.codec == "remex":
            self.q = Quantizer(self.dim, bits=self.bits, seed=self.seed, rotation=self.rotation)
        elif self.codec == "remax":
            self.q = StackedSignBitQuantizer(self.dim, self.k, seed=self.seed, rotation=self.rotation)

    @property
    def name(self) -> str:
        if self.codec == "fp32":
            return f"fp32 d={self.dim}"
        if self.codec == "remex":
            return f"remex {self.bits}-bit d={self.dim}"
        return f"remax k={self.k} d={self.dim} ({self.query_mode})"

    def build(self, dense: np.ndarray) -> "DenseIndex":
        X = mrl_truncate(dense, self.dim)
        self.n = X.shape[0]
        if self.codec == "fp32":
            self.X = X
            self.bytes_per_vec = self.dim * 4
        elif self.codec == "remex":
            self.cv = self.q.encode(X)
            # decoded, rotated, float32 — the scan representation remex's own search() caches
            self.X = self.q._get_x_hat_rot(self.cv) * self.cv.norms[:, None]
            self.bytes_per_vec = self.dim * self.bits / 8          # direction only; norms are all 1.0 here
            self.bytes_per_vec_stored = self.cv.nbytes / self.n    # what remex actually stores (+4 B norm)
        else:
            self.codes = self.q.encode(X)                          # (n, k*dim/8) uint8
            self.X = signs_pm1(self.codes, self.k * self.dim)       # +/-1 in each stack's rotated space
            self.bytes_per_vec = self.k * self.dim / 8
        return self

    def rotate_query(self, Q: np.ndarray) -> np.ndarray:
        Q = mrl_truncate(Q, self.dim)
        if self.codec == "fp32":
            return Q
        if self.codec == "remex":
            return Q @ self.q.R.T
        Qr = Q @ self.q._rotation_matrix                            # (m, k*dim)
        if self.query_mode == "sym":
            Qr = np.where(Qr > 0, 1.0, -1.0).astype(np.float32)
        return Qr

    def scores(self, Q: np.ndarray) -> np.ndarray:
        """(m, n) similarity scores; higher is better. Ranking-equivalent to cosine for fp32."""
        return self.rotate_query(Q) @ self.X.T


# --------------------------------------------------------------------------- multi-vector head
def pool_tokens(tokens: np.ndarray, offsets: np.ndarray, pool_factor: int, num_protected_tokens: int = 1):
    """Sentence Transformers' HierarchicalTokenPooling, applied to a flat token store.

    Returns (pooled_tokens float32, pooled_offsets). Cluster means are NOT
    renormalised (matches ST >= 6 / PyLate > 1.3.4).
    """
    import torch
    from sentence_transformers.multi_vector_encoder.modules.token_pooling import _hierarchical_pool_one
    out, lens = [], []
    for i in range(len(offsets) - 1):
        t = torch.from_numpy(np.ascontiguousarray(tokens[offsets[i]:offsets[i + 1]], dtype=np.float32))
        p = _hierarchical_pool_one(t, pool_factor, num_protected_tokens).numpy()
        out.append(p); lens.append(len(p))
    return np.concatenate(out), np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


@dataclass
class MultiVectorIndex:
    """Token-level index for the late-interaction head, scored with MeanMaxSim.

    codec: "fp32" | "remex" | "remax"; bits/k/query_mode as in DenseIndex.
    pool_factor: HierarchicalTokenPooling factor applied before coding (1 = off).
    """
    codec: Literal["fp32", "remex", "remax"]
    bits: int = 2
    k: int = 1
    query_mode: Literal["asym", "sym"] = "asym"
    pool_factor: int = 1
    seed: int = 0
    rotation: str = "rht"
    d: int = 128

    def __post_init__(self):
        self.q = None
        if self.codec == "remex":
            self.q = Quantizer(self.d, bits=self.bits, seed=self.seed, rotation=self.rotation)
        elif self.codec == "remax":
            self.q = StackedSignBitQuantizer(self.d, self.k, seed=self.seed, rotation=self.rotation)

    @property
    def name(self) -> str:
        pf = f" pool{self.pool_factor}" if self.pool_factor > 1 else ""
        if self.codec == "fp32":
            return f"late fp32{pf}"
        if self.codec == "remex":
            return f"late remex {self.bits}-bit{pf}"
        return f"late remax k={self.k} ({self.query_mode}){pf}"

    def build(self, tokens: np.ndarray, offsets: np.ndarray) -> "MultiVectorIndex":
        tokens = np.asarray(tokens, dtype=np.float32)
        if self.pool_factor > 1:
            tokens, offsets = pool_tokens(tokens, offsets, self.pool_factor)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.n_docs = len(offsets) - 1
        self.n_tokens = tokens.shape[0]
        if self.codec == "fp32":
            self.T = tokens
            self.bytes_total = self.n_tokens * self.d * 4
        elif self.codec == "remex":
            cv = self.q.encode(tokens)
            self.T = self.q._get_x_hat_rot(cv) * cv.norms[:, None]
            self.bytes_total = self.n_tokens * self.d * self.bits / 8
            self.bytes_total_stored = cv.nbytes                     # + 4 B/token norms
            self.cv = cv
        else:
            codes = self.q.encode(tokens)
            self.T = signs_pm1(codes, self.k * self.d)
            self.bytes_total = codes.nbytes
            self.codes = codes
        self.bytes_per_doc = self.bytes_total / self.n_docs
        return self

    def rotate_query(self, Qtok: np.ndarray) -> np.ndarray:
        Qtok = np.asarray(Qtok, dtype=np.float32)
        if self.codec == "fp32":
            return Qtok
        if self.codec == "remex":
            return Qtok @ self.q.R.T
        Qr = Qtok @ self.q._rotation_matrix
        if self.query_mode == "sym":
            Qr = np.where(Qr > 0, 1.0, -1.0).astype(np.float32)
        return Qr

    def scores(self, Qtok: np.ndarray, doc_subset: Optional[np.ndarray] = None) -> np.ndarray:
        """MeanMaxSim of one query (m_tokens, d) against every document -> (n_docs,).

        doc_subset: optional array of doc indices to score (rerank mode); returns
        scores aligned with doc_subset.
        """
        Qr = self.rotate_query(Qtok)
        if doc_subset is None:
            S = Qr @ self.T.T                                        # (m, n_tokens)
            return seg_max_mean(S, self.offsets)
        rows = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in doc_subset])
        lens = self.offsets[doc_subset + 1] - self.offsets[doc_subset]
        off = np.concatenate([[0], np.cumsum(lens)])
        S = Qr @ self.T[rows].T
        return seg_max_mean(S, off)


# --------------------------------------------------------------------------- metrics
def ndcg_at_k(ranked_ids: list[str], rel: set[str], k: int = 10) -> float:
    gains = [1.0 if d in rel else 0.0 for d in ranked_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_ids: list[str], rel: set[str], k: int = 10) -> float:
    return len(set(ranked_ids[:k]) & rel) / len(rel) if rel else 0.0


def overlap_at_k(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    """Fidelity: fraction of the fp32 top-k that the codec's top-k reproduces."""
    return len(set(a[:k].tolist()) & set(b[:k].tolist())) / k


def paired_bootstrap(delta: np.ndarray, n_boot: int = 5000, seed: int = 0):
    """Mean of per-query deltas with a 95% percentile CI and one-sided sign counts."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int((delta > 0).sum()), int((delta < 0).sum())


if __name__ == "__main__":
    # Self-test on synthetic data: codec paths agree with brute force; MaxSim agrees with sentence_transformers.
    rng = np.random.default_rng(0)
    D = l2n(rng.standard_normal((50, 1024)).astype(np.float32)); Qd = l2n(rng.standard_normal((3, 1024)).astype(np.float32))
    fp = DenseIndex("fp32", dim=1024).build(D)
    assert np.allclose(fp.scores(Qd), Qd @ D.T, atol=1e-5)
    rx = DenseIndex("remex", dim=256, bits=8).build(D)
    ref = mrl_truncate(Qd, 256) @ mrl_truncate(D, 256).T
    assert np.corrcoef(rx.scores(Qd).ravel(), ref.ravel())[0, 1] > 0.99, "remex 8-bit should be near-lossless"
    rm = DenseIndex("remax", dim=256, k=2).build(D)
    assert rm.X.shape == (50, 512) and set(np.unique(rm.X)) <= {-1.0, 1.0}
    assert np.corrcoef(rm.scores(Qd).ravel(), ref.ravel())[0, 1] > 0.6
    # multi-vector
    lens = rng.integers(3, 9, size=6); off = np.concatenate([[0], np.cumsum(lens)])
    T = l2n(rng.standard_normal((off[-1], 128)).astype(np.float32)); Qt = l2n(rng.standard_normal((4, 128)).astype(np.float32))
    mv = MultiVectorIndex("fp32").build(T, off)
    brute = np.array([(Qt @ T[off[i]:off[i + 1]].T).max(1).mean() for i in range(6)])
    assert np.allclose(mv.scores(Qt), brute, atol=1e-5)
    import torch
    from sentence_transformers.util import mean_maxsim
    st = mean_maxsim(torch.from_numpy(Qt)[None], torch.nn.utils.rnn.pad_sequence([torch.from_numpy(T[off[i]:off[i + 1]]) for i in range(6)], batch_first=True),
                     b_mask=torch.nn.utils.rnn.pad_sequence([torch.ones(l, dtype=torch.long) for l in lens], batch_first=True))[0].numpy()
    assert np.allclose(mv.scores(Qt), st, atol=1e-5), (mv.scores(Qt), st)
    sub = np.array([4, 1]); assert np.allclose(mv.scores(Qt, sub), brute[sub], atol=1e-5)
    mvx = MultiVectorIndex("remex", bits=8).build(T, off); assert np.corrcoef(mvx.scores(Qt), brute)[0, 1] > 0.99
    mvm = MultiVectorIndex("remax", k=1).build(T, off); assert mvm.bytes_per_doc == lens.mean() * 16
    pooled = MultiVectorIndex("fp32", pool_factor=2).build(T, off); assert pooled.n_tokens < off[-1]
    print("self-test ok: dense fp32/remex/remax, late fp32/remex/remax/pooled, MaxSim == sentence_transformers.mean_maxsim")
