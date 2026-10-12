"""Exact _Float128 (IEEE binary128) literals: decimal/hex text -> 128-bit pattern.

binary128 = 1 sign bit, 15 exponent bits (bias 16383), 112 fraction bits.
`binary128_from_text` returns an ExactLongDouble whose `.x87` field holds the
128-bit encoding (FloatLit.bits carries it to codegen and static initializers).
"""
from __future__ import annotations

from fractions import Fraction

from hcc.ldouble_lit import ExactLongDouble, _parse_text

_BIAS = 16383
_EMAX = 0x7FFF
_FRAC = 112


def encode_binary128(q: Fraction) -> int:
    """Round the rational `q` to the nearest binary128 value (ties to even);
    returns the 128-bit pattern (sign in bit 127)."""
    sign = (1 << 127) if q < 0 else 0
    a = abs(q)
    if a == 0:
        return sign
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while a < Fraction(2) ** e:
        e -= 1
    while a >= Fraction(2) ** (e + 1):
        e += 1
    if e + _BIAS >= _EMAX:                        # overflow: infinity
        return sign | (_EMAX << _FRAC)
    if e + _BIAS <= 0:                            # subnormal: fixed exponent 1 - bias
        m = round(a * Fraction(2) ** (_BIAS - 1 + _FRAC))
        return sign | m                           # m == 2**112 is the smallest normal
    m = round(a * Fraction(2) ** (_FRAC - e))     # significand in [2^112, 2^113]
    if m == 1 << (_FRAC + 1):
        m >>= 1
        e += 1
        if e + _BIAS >= _EMAX:
            return sign | (_EMAX << _FRAC)
    return sign | ((e + _BIAS) << _FRAC) | (m - (1 << _FRAC))


def binary128_from_text(body: str, is_hex: bool) -> ExactLongDouble:
    """Value of a _Float128 literal body: a float that carries the 128-bit `.x87`."""
    q = _parse_text(body, is_hex)
    v = ExactLongDouble(float(q) if abs(q) < Fraction(2) ** 1024 else
                        (float("inf") if q > 0 else float("-inf")))
    v.x87 = encode_binary128(q)
    return v


def float128_bits(v: float) -> int:
    """binary128 pattern of a Python float value (inf, nan, signed zero included)."""
    import math
    sign = (1 << 127) if math.copysign(1.0, v) < 0 else 0
    if math.isnan(v):
        return sign | (_EMAX << _FRAC) | (1 << (_FRAC - 1))     # quiet NaN
    if math.isinf(v):
        return sign | (_EMAX << _FRAC)
    return encode_binary128(Fraction(v))
