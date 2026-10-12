"""Bit-exact NaN constants for __builtin_nan*/__builtin_nans*.

GCC builds a quiet NaN (`nan`) or a signaling NaN (`nans`) whose payload comes
from the tag string. Quiet bit set for nan; cleared for nans. A signaling NaN
whose payload is zero would be an infinity, so GCC sets the bit just below the
quiet bit in that case (0x7fa00000 for float, 0x7ff4... for double).
"""
from __future__ import annotations

import re
import struct

from hcc import nodes as N
from hcc.typesys import TypeKind

_DIGITS = re.compile(r"\s*(0[xX][0-9a-fA-F]+|[0-9]+)")


def tag_payload(arg) -> int:
    """Payload value of the tag string argument (0 if absent or not a number)."""
    while isinstance(arg, N.Cast):          # array-to-pointer decay of the tag
        arg = arg.operand
    if not (isinstance(arg, N.VarRef) and getattr(arg.var, "is_string_literal", False)):
        return 0
    data = bytes(arg.var.init_data.data) if arg.var.init_data is not None else b""
    text = data.split(b"\0", 1)[0].decode("latin-1")
    m = _DIGITS.match(text)
    if m is None:
        return 0
    s = m.group(1)
    return int(s, 16) if s[:2] in ("0x", "0X") else int(s, 10)


def nan_bits(payload: int, ty, signaling: bool) -> int:
    """Exact encoding of a NaN of type `ty` (float, double or long double)."""
    if ty.kind == TypeKind.FLOAT:
        mant, quiet, exp = (1 << 22) - 1, 1 << 22, 0x7F800000
        bits = exp | (payload & mant)
        if not signaling:
            return bits | quiet
        return bits if payload & mant else bits | (quiet >> 1)
    if ty.kind == TypeKind.DOUBLE:
        mant, quiet, exp = (1 << 51) - 1, 1 << 51, 0x7FF0000000000000
        bits = exp | (payload & mant)
        if not signaling:
            return bits | quiet
        return bits if payload & mant else bits | (quiet >> 1)
    # x87 long double: explicit integer bit 63, quiet bit 62.
    mant, quiet = (1 << 62) - 1, 1 << 62
    bits = (0x7FFF << 64) | (1 << 63) | (payload & mant)
    if not signaling:
        return bits | quiet
    return bits if payload & mant else bits | (quiet >> 1)


def nan_literal(arg, ty, signaling: bool, loc) -> N.FloatLit:
    """Constant NaN literal of type `ty` for __builtin_nan*(arg)."""
    bits = nan_bits(tag_payload(arg), ty, signaling)
    if ty.kind == TypeKind.FLOAT:
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
    elif ty.kind == TypeKind.DOUBLE:
        value = struct.unpack("<d", struct.pack("<Q", bits))[0]
    else:
        value = float("nan")            # value is ignored: codegen uses the 80-bit bits
    return N.FloatLit(loc, value, ty=ty, bits=bits)

