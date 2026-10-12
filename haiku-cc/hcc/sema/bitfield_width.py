"""Narrowing of arithmetic on wide bit-fields (CONTRACT section 5).

GCC narrows the result of + - * << to the precision of a bit-field wider than
int (e.g. `unsigned long long b:40`): `s.b - 8` with s.b == 2 yields the
40-bit value 2^40 - 6. `narrow_wide_bitfield_result(op, lhs, rhs, node)` wraps
such an arithmetic node in a shl/shr pair that truncates it to the width.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.typing import add_type
from hcc.typesys import TY_INT, is_integer

NARROWING_OPS = ("+", "-", "*", "<<")


def _wide_bitfield(node) -> bool:
    if not isinstance(node, N.Member_) or not node.member.is_bitfield:
        return False
    ty = node.member.ty
    return ty.size > 4 and node.member.bit_width < ty.size * 8


def _narrowable_operand(node) -> bool:
    """GCC only narrows when every operand is a wide bit-field or an int-sized
    integer (`s.b - 8`), never a 64-bit operand (`s.b - 1ULL << 32`)."""
    if _wide_bitfield(node):
        return True
    return is_integer(node.ty) and node.ty.size <= 4


def _precision(node) -> int:
    """Bits of precision an operand contributes to the usual arithmetic
    conversions: a bit-field has its declared width, anything else its type."""
    add_type(node)
    if isinstance(node, N.Member_) and node.member.is_bitfield:
        return node.member.bit_width
    return node.ty.size * 8


def narrow_wide_bitfield_result(op, lhs, rhs, node):
    """Return `node` truncated to the precision of its widest operand, if a
    wide bit-field operand makes that narrower than the result type.

    GCC gives `unsigned long long a:33, b:40` operands a 40-bit precision
    (the larger of the two widths); a plain 64-bit operand keeps 64 bits."""
    if op not in NARROWING_OPS:
        return node
    operands = (lhs,) if op == "<<" else (lhs, rhs)
    if not any(_wide_bitfield(n) for n in operands):
        return node
    if not all(_narrowable_operand(n) for n in (lhs, rhs)):
        return node
    add_type(node)
    width = max(_precision(n) for n in operands)
    return _truncate_to(node, width)


def narrow_wide_bitfield_unary(node):
    """`-x` / `~x` on a wide bit-field yield the bit-field's precision
    (GCC: `~s.i` with `unsigned long i:33` and i == 2^33-1 is 0)."""
    if not isinstance(node, N.Unary) or node.op not in ("-", "~"):
        return node
    if not _wide_bitfield(node.operand):
        return node
    add_type(node)
    return _truncate_to(node, node.operand.member.bit_width)


def _truncate_to(node, width):
    """Wrap `node` in a shl/shr pair keeping only its low `width` bits."""
    ty = node.ty
    if width >= ty.size * 8:
        return node
    shift = ty.size * 8 - width
    loc = node.loc
    up = N.Binary(loc, "<<", node, N.IntLit(loc, shift, ty=TY_INT), ty=ty)
    return N.Binary(loc, ">>", up, N.IntLit(loc, shift, ty=TY_INT), ty=ty)
