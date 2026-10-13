"""Exact x87 encoding of a non-finite float/double literal widened to long double.

A static `long double x = __builtin_nan("")` (or `= -__builtin_inf()` of a double)
must be the 80-bit pattern of the converted value, payload included. Going through
a Python float loses the payload, and the 64-bit double bits must not be copied as
if they were x87 bits. Finite values are left to the float value path."""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T

def _strip_flonum_casts(node):
    """Unwrap implicit/explicit casts between floating types and leading unary minus.
    Returns (node, negated)."""
    neg = False
    while True:
        if isinstance(node, N.Cast) and node.operand.ty is not None and T.is_flonum(node.operand.ty) \
                and node.ty is not None and T.is_flonum(node.ty):
            node = node.operand
        elif isinstance(node, N.Unary) and node.op == "-":
            neg = not neg
            node = node.operand
        else:
            return node, neg


def widened_nonfinite_x87(node) -> bytes | None:
    """10-byte x87 encoding of a float/double inf or NaN literal (possibly cast or
    negated), with the source payload kept; None for any other expression."""
    inner, neg = _strip_flonum_casts(node)
    if not (isinstance(inner, N.FloatLit) and inner.bits is not None and inner.ty is not None):
        return None
    kind = inner.ty.kind
    if kind == T.TypeKind.DOUBLE:
        exp_bits, mant_bits, sign_pos = 11, 52, 63
    elif kind == T.TypeKind.FLOAT:
        exp_bits, mant_bits, sign_pos = 8, 23, 31
    else:
        return None
    bits = inner.bits
    exp = (bits >> mant_bits) & ((1 << exp_bits) - 1)
    if exp != (1 << exp_bits) - 1:       # finite: the value path handles it
        return None
    mant = bits & ((1 << mant_bits) - 1)
    sign = (bits >> sign_pos) & 1
    if neg:
        sign ^= 1
    # x87: explicit integer bit 63, then the source fraction moved to the top of the mantissa
    x87_mant = (1 << 63) | (mant << (63 - mant_bits))
    x87_bits = (sign << 79) | (0x7FFF << 64) | x87_mant
    return x87_bits.to_bytes(10, "little")
