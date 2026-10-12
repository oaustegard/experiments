"""GCC CRC builtins: __builtin_[rev_]crc{8,16,32}_data{8,16,32}(crc, data, poly).

The result and `crc`/`poly` have the width of the CRC (uint8/16/32_t); `data` has
the width of the data suffix. Lowered to a BuiltinCall named "[rev_]crc<N>_data<D>"
which codegen/crc.py expands inline. Being special builtins, these names must not
reach the libc-name lookup in primary.py (a user function may share the name).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.conversions import new_cast

_WIDTH_TY = {8: T.TY_UCHAR, 16: T.TY_USHORT, 32: T.TY_UINT, 64: T.TY_ULONG}

CRC_BUILTINS = frozenset(
    f"__builtin_{prefix}crc{n}_data{d}"
    for prefix in ("", "rev_")
    for n in (8, 16, 32, 64)
    for d in (8, 16, 32, 64) if d <= n
)


def crc_call(parser, tok):
    """Parse the argument list (after '(') of a CRC builtin and return the BuiltinCall."""
    name = tok.text[len("__builtin_"):]
    rest = name.split("crc", 1)[1]               # "<N>_data<D>"
    n_bits, d_bits = (int(x) for x in rest.split("_data"))
    crc, data, poly = parser._args(3)
    args = [new_cast(crc, _WIDTH_TY[n_bits]), new_cast(data, _WIDTH_TY[d_bits]),
            new_cast(poly, _WIDTH_TY[n_bits])]
    return N.BuiltinCall(tok.loc, name, args, None, ty=_WIDTH_TY[n_bits])
