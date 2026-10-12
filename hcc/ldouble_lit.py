"""Exact long double literals: decimal/hex text -> 80-bit x87 extended encoding.

A Python float cannot hold 1e4932 or 3e-4000, and a double round trip loses
long double digits. The literal scanner calls `x87_from_text` to get the exact
encoding (round to nearest, ties to even, 64-bit significand) and keeps it on
the float value as `.x87` (see `ExactLongDouble`); FloatLit.bits carries it to
codegen and static initializers.
"""

from __future__ import annotations

from fractions import Fraction

_BIAS = 16383
_EMAX = 0x7FFF


class ExactLongDouble(float):
    """A float value that also remembers its exact x87 encoding (10 bytes as int)."""
    x87: int = 0


def _parse_text(body: str, is_hex: bool) -> Fraction:
    """Exact rational value of a C floating literal body (suffix already removed)."""
    if not is_hex:
        return Fraction(body)
    low = body.lower()
    if "p" in low:
        mant, exp2 = low[2:].split("p", 1)
        e = int(exp2)
    else:
        mant, e = low[2:], 0
    if "." in mant:
        ip, fp = mant.split(".", 1)
    else:
        ip, fp = mant, ""
    digits = int((ip + fp) or "0", 16)
    return Fraction(digits) * Fraction(2) ** (e - 4 * len(fp))


def encode_x87(q: Fraction, saturate: bool = False) -> int:
    """Round the rational `q` to the nearest x87 extended value; returns the
    80-bit pattern as an int (sign in bit 79). `saturate` (used for _Float128,
    which has a 113-bit significand): a value that only rounds up past the
    largest finite x87 value gives that finite value instead of infinity."""
    sign = 0x8000 if q < 0 else 0
    a = abs(q)
    if a == 0:
        return sign
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while a < Fraction(2) ** e:
        e -= 1
    while a >= Fraction(2) ** (e + 1):
        e += 1
    if e + _BIAS >= _EMAX:                       # overflow: infinity
        return (sign | _EMAX) << 64 | (1 << 63)
    if e + _BIAS <= 0:                           # subnormal: fixed exponent 1 - bias
        m = round(a * Fraction(2) ** (_BIAS - 1 + 63))
        expf = 1 if m >= (1 << 63) else 0
        return ((sign | expf) << 64) | m
    m = round(a * Fraction(2) ** (63 - e))       # 64-bit significand in [2^63, 2^64]
    if m == 1 << 64:
        m >>= 1
        e += 1
        if e + _BIAS >= _EMAX:
            if saturate:                         # largest finite: exponent 0x7ffe, all ones
                return (sign | (_EMAX - 1)) << 64 | ((1 << 64) - 1)
            return (sign | _EMAX) << 64 | (1 << 63)
    return ((sign | (e + _BIAS)) << 64) | m


def x87_from_text(body: str, is_hex: bool, saturate: bool = False) -> ExactLongDouble:
    """Value of a long double literal body: a float that carries `.x87`."""
    q = _parse_text(body, is_hex)
    v = ExactLongDouble(float(q) if abs(q) < Fraction(2) ** 1024 else
                        (float("inf") if q > 0 else float("-inf")))
    v.x87 = encode_x87(q, saturate)
    return v
