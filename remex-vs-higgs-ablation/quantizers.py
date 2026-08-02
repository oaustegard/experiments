#!/usr/bin/env python3
"""The 2x2x2 factorial: rotation x norm handling x codebook.

    axis A  rotation   Haar (dense, O(d^2))      vs  randomized Hadamard (O(d log d))
    axis B  norm       exact fp32 norm, out-of-band  vs  per-block scale in the payload
    axis C  codebook   scalar Lloyd-Max          vs  Gaussian-optimal m-dim grid

remex   = (haar, exactnorm, scalar)
HIGGS-like = (rht,  blockscale, vector)

The six remaining cells are the interaction terms, which is the whole point:
if remex loses to HIGGS-like it matters enormously *which* axis carries the
loss, and only the full factorial separates them.

Bit accounting is done in `bytes_per_vector()` and includes every side
channel — the fp32 norm, the fp16 block scales, and a note on the amortized
codebook.  Nominal bits/coordinate is not a budget.

Randomized Hadamard without padding
-----------------------------------
The textbook RHT needs d to be a power of two.  Padding d=768 to 1024 would
hand the RHT arm a 33% larger payload and quietly destroy the matched budget,
so instead this uses rounds of (random permutation -> block-diagonal
randomized Hadamard) with block size = the largest power of two dividing d,
enough rounds to mix every coordinate.  Still O(d log d), still orthogonal,
and exactly d coordinates out.  `calibrate.py` checks its incoherence against
the Haar rotation rather than assuming it.
"""
from __future__ import annotations

import math

import numpy as np

from grids import build_codebook

BLOCK = 128  # per-block scale group size for the axis-B "blockscale" arm


# --------------------------------------------------------------------------
# axis A — rotations


class HaarRotation:
    """Dense random orthogonal matrix, Haar-distributed.  O(d^2) to apply."""

    name = "haar"

    def __init__(self, d: int, seed: int):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((d, d))
        q, r = np.linalg.qr(a)
        # sign-correct the QR so Q is Haar rather than merely orthogonal
        q *= np.sign(np.diag(r))
        self.Q = np.ascontiguousarray(q.astype(np.float32))
        self.d = d

    def apply(self, X):
        return X @ self.Q

    def inverse(self, Y):
        return Y @ self.Q.T


_HADAMARD_CACHE: dict[int, np.ndarray] = {}

# Above this block size the explicit matrix stops paying: the butterfly does
# log2(B) passes over the data while the matmul does B multiply-adds per
# element, so BLAS's constant-factor advantage is eventually swamped.
# Measured crossover on this container is B ~ 2048; 1024 is the conservative
# side of it.  See RESULTS.md "Axis A".
FWHT_MATMUL_MAX_B = 1024


def _hadamard(B: int) -> np.ndarray:
    """Orthonormal Sylvester Hadamard matrix, cached per block size."""
    H = _HADAMARD_CACHE.get(B)
    if H is None:
        from scipy.linalg import hadamard as _sc_hadamard

        H = np.ascontiguousarray(_sc_hadamard(B).astype(np.float32) / math.sqrt(B))
        _HADAMARD_CACHE[B] = H
    return H


def fwht(a: np.ndarray) -> np.ndarray:
    """Normalized fast Walsh-Hadamard transform along the last axis.

    Normalized by 1/sqrt(B), so the transform is orthogonal *and* symmetric --
    it is its own inverse, which is why `RHTRotation.inverse` can reuse it.

    Two implementations, selected by block size, because the honest comparison
    against a dense BLAS rotation needs the fast transform to actually be fast:

      * B <= FWHT_MATMUL_MAX_B: one `sgemm` against a cached Hadamard matrix.
        More arithmetic than the butterfly, but it runs at BLAS speed instead
        of at numpy-elementwise speed, and it is what wins at every dimension
        anyone builds a retrieval index at.
      * larger B: a copy-free butterfly that ping-pongs between two buffers.
        `np.add(..., out=)` into the destination avoids the two full-array
        temporaries per stage that the naive in-place version needs.

    Both agree with the naive butterfly exactly (matmul to float32 rounding).
    """
    B = a.shape[-1]
    lead = a.shape[:-1]
    if B <= FWHT_MATMUL_MAX_B:
        flat = np.ascontiguousarray(a, dtype=np.float32).reshape(-1, B)
        return (flat @ _hadamard(B)).reshape(*lead, B)
    # np.array(copy) is required: the ping-pong can end on the input buffer,
    # and the final in-place scaling would then mutate the caller's array.
    src = np.array(a, dtype=np.float32, order="C").reshape(-1, B)
    dst = np.empty_like(src)
    h = 1
    while h < B:
        s = src.reshape(-1, B // (2 * h), 2, h)
        d = dst.reshape(-1, B // (2 * h), 2, h)
        np.add(s[:, :, 0, :], s[:, :, 1, :], out=d[:, :, 0, :])
        np.subtract(s[:, :, 0, :], s[:, :, 1, :], out=d[:, :, 1, :])
        src, dst = dst, src
        h *= 2
    src = src.reshape(-1, B)
    src /= math.sqrt(B)
    return src.reshape(*lead, B)


def _largest_pow2_divisor(d: int) -> int:
    b = 1
    while d % (b * 2) == 0:
        b *= 2
    return b


class RHTRotation:
    """Randomized Hadamard: rounds of (permute, sign-flip, block-diagonal FWHT).

    One round on a single block is the classical RHT.  Multiple rounds exist
    only so that dimensions like 768 and 100, whose largest power-of-two
    divisor is smaller than d, still mix across the whole vector without
    padding.
    """

    name = "rht"

    def __init__(self, d: int, seed: int):
        rng = np.random.default_rng(seed)
        self.d = d
        self.B = _largest_pow2_divisor(d)
        if self.B < 2:
            raise ValueError(f"d={d} has no usable Hadamard block")
        rounds = 1 if self.B == d else max(2, math.ceil(math.log(d) / math.log(self.B)))
        # The classical RHT is H.D -- a sign flip then a full-width Hadamard.
        # The permutation exists only to mix ACROSS blocks when B < d; with a
        # single full-width block it is dead weight, and a gather over a
        # (n, d) array is the most expensive step in the whole transform.
        self.permute = not (rounds == 1 and self.B == d)
        self.perms = [rng.permutation(d) for _ in range(rounds)] if self.permute else []
        self.inv_perms = [np.argsort(p) for p in self.perms]
        self.signs = [rng.choice(np.array([-1.0, 1.0], np.float32), size=d)
                      for _ in range(rounds)]

    def apply(self, X):
        Y = np.ascontiguousarray(X, dtype=np.float32)
        n = Y.shape[0]
        perms = self.perms if self.permute else [None] * len(self.signs)
        for perm, sign in zip(perms, self.signs):
            Y = np.take(Y, perm, axis=1) * sign if perm is not None else Y * sign
            Y = fwht(Y.reshape(n, self.d // self.B, self.B)).reshape(n, self.d)
        return Y

    def inverse(self, Y):
        Z = np.ascontiguousarray(Y, dtype=np.float32)
        n = Z.shape[0]
        inv = self.inv_perms if self.permute else [None] * len(self.signs)
        for perm_inv, sign in zip(reversed(inv), reversed(self.signs)):
            Z = fwht(Z.reshape(n, self.d // self.B, self.B)).reshape(n, self.d)
            Z = Z * sign
            if perm_inv is not None:
                Z = np.take(Z, perm_inv, axis=1)
        return Z


class IdentityRotation:
    name = "none"

    def __init__(self, d: int, seed: int):
        self.d = d

    def apply(self, X):
        return np.ascontiguousarray(X, dtype=np.float32)

    def inverse(self, Y):
        return np.ascontiguousarray(Y, dtype=np.float32)


ROTATIONS = {"haar": HaarRotation, "rht": RHTRotation, "none": IdentityRotation}


# --------------------------------------------------------------------------
# the arms


class Arm:
    """One cell of the factorial.

    encode/decode round-trips a matrix and reports its own honest byte cost.
    Everything is data-oblivious: the rotation comes from a seed, the codebook
    from the standard normal.  Nothing is fitted to the corpus, in either arm.
    """

    def __init__(self, *, rotation: str, norm: str, codebook: str, bits: int,
                 d: int, seed: int, log=None):
        self.rotation_kind = rotation
        self.norm_kind = norm            # "exactnorm" | "blockscale"
        self.codebook_kind = codebook    # "scalar" | "vector" | "uniform"
        self.bits = bits
        self.d = d
        self.seed = seed
        self.R = ROTATIONS[rotation](d, seed)
        self.cb = build_codebook(codebook, bits, d, log=log)
        # Largest divisor of d that is <= BLOCK and still leaves >= 2 groups.
        # The naive min(BLOCK, d) collapses d=100 to a single block, which
        # turns the axis-B "per-block scale" arm into a global scale and stops
        # the axis from testing block granularity at all on that corpus --
        # exactly the degeneracy that would make axis B look null for the
        # wrong reason.  d=100 -> 50 (2 groups), 768 -> 128 (6), 1024 -> 128 (8).
        cap = min(BLOCK, d // 2) or 1
        sub = self.cb.m
        # The block boundary must not cut a sub-vector in half.  If it does,
        # some sub-vectors get their two halves divided by DIFFERENT fp16
        # scales before hitting a grid trained on N(0, I_m), which corrupts
        # only the blockscale+vector cell -- i.e. the HIGGS-like arm -- and so
        # confounds axes B and C rather than degrading anything uniformly.
        # Concretely this bit d=100 at 4 bits, where block=50 and m=4 left
        # 2 of every 25 sub-vectors straddling a boundary.
        ok = [k for k in range(1, cap + 1) if d % k == 0 and k % sub == 0]
        self.block = max(ok) if ok else max(k for k in range(1, cap + 1) if d % k == 0)
        if self.block % sub:
            raise ValueError(f"block {self.block} does not tile sub-vector dim {sub} at d={d}")
        self.nblocks = d // self.block

    # -- naming -----------------------------------------------------------
    @property
    def label(self):
        return f"{self.rotation_kind}+{self.norm_kind}+{self.codebook_kind}"

    # -- byte accounting --------------------------------------------------
    def bytes_per_vector(self) -> dict:
        """Every byte that has to be stored per vector, itemised.

        The rotation and the codebook are excluded because they are shared
        across the whole index (one d x d matrix or one K x m grid for
        millions of vectors), the same way HIGGS and remex both exclude them.
        That exclusion is stated rather than assumed — see RESULTS.md.
        """
        payload = self.d * self.bits / 8.0
        side = 4.0 if self.norm_kind == "exactnorm" else 2.0 * self.nblocks
        return {"payload": payload, "side": side, "total": payload + side}

    def shared_bytes(self) -> int:
        """Index-level amortized cost: rotation + codebook."""
        rot = 0
        if self.rotation_kind == "haar":
            rot = self.d * self.d * 4
        elif self.rotation_kind == "rht":
            rot = len(self.R.perms) * self.d * (4 + 1)  # perm int32 + sign int8
        cb = getattr(self.cb, "C", None)
        cbb = int(cb.size * 4) if cb is not None else int(self.cb.levels.size * 4)
        return rot + cbb

    # -- the codec --------------------------------------------------------
    def encode_decode(self, X: np.ndarray) -> np.ndarray:
        """Round-trip X (n, d) through the arm, returning the reconstruction.

        Both arms are evaluated by reconstruction rather than by an
        asymmetric code-domain score, so that cosine and inner product are
        computed the same way for every arm and the comparison cannot be
        confounded by a scoring shortcut available to only one of them.
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        if self.norm_kind == "exactnorm":
            nrm = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
            U = X / np.maximum(nrm, 1e-12)
            Y = self.R.apply(U)
            # a rotated unit vector has coordinate variance exactly 1/d; this
            # is a constant, not a fitted statistic
            sigma = np.float32(1.0 / math.sqrt(d))
            Yq = self.cb.encode_decode(Y / sigma) * sigma
            return self.R.inverse(Yq) * nrm
        # blockscale: fold the scale into the payload, per group of `block`
        Y = self.R.apply(X)
        Yb = Y.reshape(n, self.nblocks, self.block)
        scale = np.sqrt(np.mean(Yb.astype(np.float32) ** 2, axis=2, keepdims=True))
        scale = np.maximum(scale, 1e-12).astype(np.float16).astype(np.float32)
        Yq = self.cb.encode_decode((Yb / scale).reshape(n, d)).reshape(
            n, self.nblocks, self.block) * scale
        return self.R.inverse(Yq.reshape(n, d))


class QJLArm(Arm):
    """TurboQuant `prod` replication control: scalar Lloyd-Max + 1-bit QJL
    residual, at a matched *total* budget of `bits` (so LM gets bits-1).

    Carried only because the issue asks for the 2026-04-02 result to be
    replicated, not re-litigated: `prod` was strictly dominated at every bit
    width.  If this arm comes out competitive here, the harness is wrong.
    """

    def __init__(self, *, bits: int, d: int, seed: int, log=None, **kw):
        super().__init__(rotation="haar", norm="exactnorm", codebook="scalar",
                         bits=max(1, bits - 1), d=d, seed=seed, log=log)
        self.total_bits = bits
        rng = np.random.default_rng(seed + 99991)
        self.S = rng.standard_normal((d, d)).astype(np.float32) / math.sqrt(d)

    @property
    def label(self):
        return "control:lm+qjl"

    def bytes_per_vector(self):
        # 4 B for the vector norm, plus 4 B for the RESIDUAL norm: the QJL
        # decoder below scales the sign sketch by ||resid||, which is a real
        # per-vector quantity a decoder cannot recompute from the code.
        # Charging only the first would have let this control cheat by 4 B.
        payload = self.d * self.total_bits / 8.0
        return {"payload": payload, "side": 8.0, "total": payload + 8.0}

    def shared_bytes(self):
        # the d x d JL sketch is shared across the index, like the rotation
        return super().shared_bytes() + self.d * self.d * 4

    def encode_decode(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        d = X.shape[1]
        nrm = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
        U = X / np.maximum(nrm, 1e-12)
        Y = self.R.apply(U)
        sigma = np.float32(1.0 / math.sqrt(d))
        Yq = self.cb.encode_decode(Y / sigma) * sigma
        resid = Y - Yq
        # 1-bit QJL: keep only sign(S r); the unbiased inverse scales by
        # sqrt(pi/2) * ||r|| / d over the d sign bits
        sgn = np.sign(resid @ self.S.T).astype(np.float32)
        rn = np.linalg.norm(resid, axis=1, keepdims=True)
        est = (sgn @ self.S) * (math.sqrt(math.pi / 2) * rn / d)
        return self.R.inverse(Yq + est) * nrm


def make_arm(spec: dict, bits: int, d: int, seed: int, log=None) -> Arm:
    if spec.get("control") == "qjl":
        return QJLArm(bits=bits, d=d, seed=seed, log=log)
    return Arm(rotation=spec["rotation"], norm=spec["norm"],
               codebook=spec["codebook"], bits=bits, d=d, seed=seed, log=log)


#: The full factorial plus the two quantized controls.  fp32 is handled
#: separately by the runner since it has no bit width.
FACTORIAL = [
    {"rotation": r, "norm": nm, "codebook": c}
    for r in ("haar", "rht")
    for nm in ("exactnorm", "blockscale")
    for c in ("scalar", "vector")
]

CONTROLS = [
    {"rotation": "none", "norm": "blockscale", "codebook": "uniform"},  # floor
    {"control": "qjl"},                                                 # replication
]
