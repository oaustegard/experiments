"""Constant folding of integer -> floating conversions (C 6.3.1.4).

The value is rounded once, from the exact integer, to the target format: a
Python float round trip would round twice (e.g. 16777217 -> double -> float
is exact in double but must round to 16777216 in float, and a 64-bit integer
rounded to double and then to float can land on the wrong neighbour)."""

from hcc.typesys import TypeKind


def int_to_flonum(v: int, ty) -> float:
    """The integer `v` converted to float type `ty`, as a Python float that
    holds the exact value of the target format."""
    if ty.kind == TypeKind.FLOAT and abs(v) >= 1 << 24:
        neg, m = v < 0, abs(v)
        shift = m.bit_length() - 24           # keep 24 significant bits
        q, r = m >> shift, m & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if r > half or (r == half and q & 1):  # round half to even
            q += 1
        m = q << shift
        v = -m if neg else m
    return float(v)
