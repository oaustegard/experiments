"""rhq: the table codec the WGSL engine decodes. Per row: one fp16 scale and b-bit codes of the row
after a fixed randomized Hadamard rotation, on Lloyd-Max levels for N(0, 1). Same shape as remex
(rotation + Lloyd-Max + one scalar per row), with a rotation that is cheap on a GPU: z = H(s * x),
H the orthonormal 512-point Walsh-Hadamard transform (symmetric, its own inverse), s fixed signs.

decode(row) = scale * s * H(levels[codes])   (H applied to the 512 level values)
The scale is the least-squares fit <x, x_hat>/|x_hat|^2 against the unit-scale decode.
Codes are packed little-endian in a row-wise bitstream: row r of a b-bit tier is 64*b bytes.
"""
import numpy as np

D = 512


def signs(seed=0x5EED):
    """512 signs from a 32-bit xorshift (reproducible in JS/WGSL without a library)."""
    x, out = seed & 0xFFFFFFFF, np.empty(D, np.float32)
    for i in range(D):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = 1.0 if x & 1 else -1.0
    return out


def fwht(X):
    """Orthonormal Walsh-Hadamard along the last axis (natural order)."""
    X = np.array(X, np.float32, copy=True)
    h = 1
    while h < D:
        X = X.reshape(*X.shape[:-1], D // (2 * h), 2, h)
        a, b = X[..., 0, :].copy(), X[..., 1, :].copy()
        X[..., 0, :], X[..., 1, :] = a + b, a - b
        X = X.reshape(*X.shape[:-3], D)
        h *= 2
    return X / np.float32(np.sqrt(D))


def lloyd_max(bits, iters=200):
    """Lloyd-Max levels for N(0,1), computed on a fine grid."""
    from scipy.stats import norm
    k = 2 ** bits
    c = norm.ppf((np.arange(k) + 0.5) / k)
    for _ in range(iters):
        t = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
        num = norm.pdf(t[:-1]) - norm.pdf(t[1:])
        den = norm.cdf(t[1:]) - norm.cdf(t[:-1])
        c = num / den
    return c.astype(np.float32)


def encode(X, bits, s=None):
    s = signs() if s is None else s
    X = np.asarray(X, np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    Z = fwht(X * s) / np.maximum(n, 1e-12) * np.float32(np.sqrt(D))  # ~N(0,1) per component
    lv = lloyd_max(bits)
    codes = np.abs(Z[..., None] - lv).argmin(-1).astype(np.uint8)
    Xh = fwht(lv[codes]) * s
    scale = ((X * Xh).sum(1) / np.maximum((Xh * Xh).sum(1), 1e-12)).astype(np.float16)
    return codes, scale


def decode(codes, scale, bits, s=None):
    s = signs() if s is None else s
    return fwht(lloyd_max(bits)[codes]) * s * scale.astype(np.float32)[:, None]


def pack(codes, bits):
    """(n, 512) codes -> (n, 64*bits) bytes, little-endian bitstream per row."""
    n = len(codes)
    bitsarr = ((codes[..., None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(n, D * bits)
    return np.packbits(bitsarr, axis=1, bitorder="little")


def unpack(packed, bits):
    n = len(packed)
    b = np.unpackbits(packed, axis=1, bitorder="little")[:, :D * bits].reshape(n, D, bits)
    return (b << np.arange(bits, dtype=np.uint8)).sum(-1).astype(np.uint8)
