"""IEEE 754-2008 BID encoding of decimal floating constants (_Decimal32/64/128).

    bid_from_int(v, size) -> bytes   little-endian BID bits of integer v in a
                                     decimal format of `size` bytes (4, 8, 16)

Integers that need more digits than the format holds are rounded to nearest,
ties to even, with the exponent raised (as an IEEE conversion would).
"""
from __future__ import annotations

# size: (decimal digits, exponent bias, coefficient bits, exponent bits)
_FORMATS = {4: (7, 101, 23, 8), 8: (16, 398, 53, 10), 16: (34, 6176, 113, 14)}


def bid_from_int(v: int, size: int) -> bytes:
    digits, bias, cbits, _ebits = _FORMATS[size]
    neg = 1 if v < 0 else 0
    c = abs(v)
    q = 0
    limit = 10 ** digits
    while c >= limit:
        n = len(str(c)) - digits
        d = 10 ** n
        qq, r = divmod(c, d)
        if 2 * r > d or (2 * r == d and qq % 2 == 1):
            qq += 1
        c, q = qq, q + n
        if c == limit:              # rounding carried into a new digit
            c //= 10
            q += 1
    e = q + bias
    top = size * 8 - 1
    if c < (1 << cbits):            # small form: exponent then full coefficient
        bits = (neg << top) | (e << cbits) | c
    else:                           # large form: 11 prefix, exponent, implicit 100 coefficient
        bits = (neg << top) | (3 << (top - 2)) | (e << (cbits - 2)) | (c - (1 << cbits))
    return bits.to_bytes(size, "little")
