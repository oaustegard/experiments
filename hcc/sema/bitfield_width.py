"""Narrowing of arithmetic on wide bit-fields (CONTRACT section 5).

GCC narrows the result of + - * << to the precision of a bit-field wider than
int (e.g. `unsigned long long b:40`): `s.b - 8` with s.b == 2 yields the
40-bit value 2^40 - 6. `narrow_wide_bitfield_result(op, lhs, rhs, node)` wraps
such an arithmetic node in a shl/shr pair that truncates it to the width.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.typing import add_type
from hcc.typesys import TY_INT

NARROWING_OPS = ("+", "-", "*", "<<")


def _wide_bitfield(node) -> bool:
    if not isinstance(node, N.Member_) or not node.member.is_bitfield:
        return False
    ty = node.member.ty
    return ty.size > 4 and node.member.bit_width < ty.size * 8


def narrow_wide_bitfield_result(op, lhs, rhs, node):
    """Return `node` truncated to the narrowest wide bit-field operand's width."""
    if op not in NARROWING_OPS:
        return node
    widths = [n.member.bit_width for n in (lhs, rhs) if _wide_bitfield(n)]
    if not widths:
        return node
    add_type(node)
    ty = node.ty
    shift = ty.size * 8 - min(widths)
    loc = node.loc
    up = N.Binary(loc, "<<", node, N.IntLit(loc, shift, ty=TY_INT), ty=ty)
    op2 = ">>"
    return N.Binary(loc, op2, up, N.IntLit(loc, shift, ty=TY_INT), ty=ty)
