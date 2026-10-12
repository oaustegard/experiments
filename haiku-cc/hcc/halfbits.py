"""IEEE binary16 (_Float16) bit-level conversions, shared by the parser, sema and codegen.

Python's struct '<e' format does not keep NaN payloads / signaling bits, so NaNs
are converted by hand: the top 10 payload bits are kept, and a payload that would
become zero is set to 1 so a signaling NaN stays a NaN (GCC keeps sNaN too).
"""
from __future__ import annotations

import math
import struct


def half_bits(v: float) -> int:
    """Nearest binary16 encoding of the Python float `v` (ties to even, overflow -> inf)."""
    if v != v:
        bits = struct.unpack("<Q", struct.pack("<d", v))[0]
        payload = (bits >> 42) & 0x3FF
        if payload == 0 and bits & ((1 << 42) - 1):
            payload = 1
        return ((bits >> 48) & 0x8000) | 0x7C00 | payload
    try:
        return struct.unpack("<H", struct.pack("<e", v))[0]
    except OverflowError:               # magnitude rounds to infinity
        return 0x7C00 | (0x8000 if math.copysign(1.0, v) < 0 else 0)


def half_to_double(h: int) -> float:
    """Exact Python float value of a binary16 encoding (NaN payloads preserved)."""
    if (h & 0x7C00) == 0x7C00 and h & 0x3FF:
        bits = ((h & 0x8000) << 48) | (0x7FF << 52) | ((h & 0x3FF) << 42)
        return struct.unpack("<d", struct.pack("<Q", bits))[0]
    return struct.unpack("<e", struct.pack("<H", h))[0]


def round_to_half(v: float) -> float:
    """Round a Python float to the nearest binary16 value (ties to even)."""
    return half_to_double(half_bits(v))


def half_to_float_bits(h: int) -> int:
    """float32 encoding of a binary16 encoding (exact; sNaN stays a NaN)."""
    sign = (h & 0x8000) << 16
    exp, mant = (h >> 10) & 0x1F, h & 0x3FF
    if exp == 0x1F:
        return sign | 0x7F800000 | (mant << 13)
    return struct.unpack("<I", struct.pack("<f", half_to_double(h)))[0]
