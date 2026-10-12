"""C23 `constexpr` object initializer checks.

A constexpr object's initializer values must be representable in the type of the
object (C23 6.7.1p?; GCC: "'constexpr' initializer not representable in type of
object"). Only constant values are checked; non-constant initializers are left to
the normal conversion rules.

    check_scalar(parser, value, ty)      integer constant `value` into integer type `ty`
    check_string_units(parser, units, src_unsigned, ty)
                                         string literal code units (raw, zero-extended)
                                         read as the source char type into element `ty`
"""
from __future__ import annotations

from hcc import typesys as T

MSG = "'constexpr' initializer not representable in type of object"


def representable(value: int, ty) -> bool:
    bits = ty.size * 8
    if ty.is_unsigned or ty.kind == T.TypeKind.BOOL:
        return 0 <= value < (1 << bits)
    return -(1 << (bits - 1)) <= value < (1 << (bits - 1))


def check_scalar(parser, value, ty) -> None:
    if value is None or not T.is_integer(ty) or ty.kind == T.TypeKind.BOOL:
        return
    if not representable(value, ty):
        parser.error(MSG)


def check_string_units(parser, units, src_bits: int, src_unsigned: bool, ty) -> None:
    """Each code unit is first read as a value of the source character type (plain
    strings: `char`, u8 strings: `unsigned char`), then must fit the element type."""
    if not T.is_integer(ty) or ty.kind == T.TypeKind.BOOL:
        return
    for v in units:
        if not src_unsigned and v >= 1 << (src_bits - 1):
            v -= 1 << src_bits
        if not representable(v, ty):
            parser.error(MSG)
            return
