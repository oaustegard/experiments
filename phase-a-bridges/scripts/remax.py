"""Remax = stacked SimHash quantizer.

For each input vector v in R^d, draw k random hyperplanes per output bit
position and pack the resulting d*k sign bits into k bytes per dimension.
At k=2 this gives 2*d = 1536 bits per vector (192 bytes). Hamming distance
between two packed codes correlates with cosine distance between the
originals (the more bits, the tighter; k=2 is the elbow per phase-0).

This is a minimal, dependency-free implementation tuned for the phase A
MVP. Not optimised for production scale (that's phase B).
"""
from __future__ import annotations

import numpy as np


class RemaxBuilder:
    def __init__(self, d: int, k: int = 2, *, seed: int = 0, n_estimated: int = 1000):
        self.d = d
        self.k = k
        self.seed = seed
        rng = np.random.default_rng(seed)
        # Random Gaussian projection matrix: shape (d, d*k)
        # Each column is a random hyperplane normal.
        self.R = rng.standard_normal((d, d * k), dtype=np.float32)
        self.codes: list[np.ndarray] = []
        self.ids: list[str] = []

    @property
    def bits_per_vec(self) -> int:
        return self.d * self.k

    @property
    def bytes_per_vec(self) -> int:
        return (self.bits_per_vec + 7) // 8

    def project_and_pack(self, vecs: np.ndarray) -> np.ndarray:
        """vecs: (n, d) float -> (n, bytes_per_vec) uint8 packed bits."""
        # SimHash: sign of projection -> 1 if >=0 else 0
        signs = (vecs @ self.R) >= 0  # (n, d*k) bool
        # packbits returns big-endian within each byte; that's fine as long as
        # we use the same orientation everywhere.
        return np.packbits(signs.astype(np.uint8), axis=1)

    def append(self, ids: list[str], packed: np.ndarray) -> None:
        assert len(ids) == packed.shape[0]
        self.ids.extend(ids)
        self.codes.append(packed)

    def partial_codes(self) -> np.ndarray:
        if not self.codes:
            return np.zeros((0, self.bytes_per_vec), dtype=np.uint8)
        return np.concatenate(self.codes, axis=0)


# ---------------------------------------------------------------------------
# Distance computations
# ---------------------------------------------------------------------------

# Precomputed popcount table for uint8.
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between rows of A and B (packed uint8)."""
    assert A.dtype == np.uint8 and B.dtype == np.uint8
    assert A.shape[1] == B.shape[1]
    # XOR via broadcasting: (n, 1, bytes) ^ (1, m, bytes) -> (n, m, bytes)
    xor = A[:, None, :] ^ B[None, :, :]
    return _POPCOUNT8[xor].sum(axis=2).astype(np.int32)


def hamming_pairs_chunked(
    A: np.ndarray, B: np.ndarray, *, chunk: int = 64
) -> np.ndarray:
    """Memory-safe version: row-chunks A. Same result, less peak RAM."""
    n, m = A.shape[0], B.shape[0]
    out = np.empty((n, m), dtype=np.int32)
    for i in range(0, n, chunk):
        out[i:i + chunk] = hamming_pairwise(A[i:i + chunk], B)
    return out


def normalized_hamming(d: np.ndarray, bits: int) -> np.ndarray:
    return d.astype(np.float32) / float(bits)
