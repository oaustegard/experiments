"""IEEE-754 encoding helpers for constant floating values.

pack_float32 / pack_float64 return the little-endian bytes of a Python float
rounded to the target format. Values that round to infinity in the target
format (e.g. 3.5e38f, which exceeds FLT_MAX) become a signed infinity, as
C requires, instead of raising OverflowError from struct.pack.
"""
from __future__ import annotations

import math
import struct


def pack_float32(v: float) -> bytes:
    try:
        return struct.pack("<f", v)
    except OverflowError:
        return struct.pack("<f", math.copysign(math.inf, v))


def pack_float64(v: float) -> bytes:
    try:
        return struct.pack("<d", v)
    except OverflowError:
        return struct.pack("<d", math.copysign(math.inf, v))
