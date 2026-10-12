"""Scalar operands of binary operators applied to GCC vectors.

In `v op s` (or `s op v`) with `v` a vector and `s` an arithmetic scalar, C
converts `s` to the element type and applies `op` element-wise. A vector value
is an address (see vector_cast.py), so the scalar is broadcast into an
anonymous vector local:
    (tmp_s = (E)s, tmp[0] = tmp_s, ..., tmp[n-1] = tmp_s, tmp)
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.arith import new_add
from hcc.sema.conversions import new_cast
from hcc.typesys import TY_INT, is_numeric, pointer_to

VECTOR_BINOPS = ("+", "-", "*", "/", "%", "&", "|", "^")


def broadcast_scalar(parser, loc, scalar: N.Expr, vec_ty) -> N.Expr:
    elem = vec_ty.base
    count = vec_ty.size // elem.size
    tmp = parser.new_temp(vec_ty, loc)
    sc = parser.new_temp(elem, loc)
    value = parser.typed(N.Assign(loc, parser.typed(N.VarRef(loc, sc)),
                                  new_cast(parser.typed(scalar), elem)))
    addr = parser.typed(N.AddrOf(loc, parser.typed(N.VarRef(loc, tmp))))
    base = parser.typed(N.Cast(loc, addr, ty=pointer_to(elem)))
    result = parser.typed(N.VarRef(loc, tmp))
    for i in reversed(range(count)):      # built back to front: store_0 runs first
        ptr = new_add(base, N.IntLit(loc, i, ty=TY_INT), loc)
        slot = parser.typed(N.Deref(loc, parser.typed(ptr)))
        store = parser.typed(N.Assign(loc, slot, parser.typed(N.VarRef(loc, sc))))
        result = parser.typed(N.Comma(loc, store, result))
    return parser.typed(N.Comma(loc, value, result))


def promote_vector_operands(parser, loc, op: str, lhs: N.Expr, rhs: N.Expr):
    """Return (lhs, rhs) with a scalar operand of a vector binop broadcast."""
    if op not in VECTOR_BINOPS:
        return lhs, rhs
    lt, rt = parser.typed(lhs).ty, parser.typed(rhs).ty
    if lt.is_vector and not rt.is_vector and is_numeric(rt):
        rhs = broadcast_scalar(parser, loc, rhs, lt)
    elif rt.is_vector and not lt.is_vector and is_numeric(lt):
        lhs = broadcast_scalar(parser, loc, lhs, rt)
    return lhs, rhs


__all__ = ["promote_vector_operands", "broadcast_scalar"]
