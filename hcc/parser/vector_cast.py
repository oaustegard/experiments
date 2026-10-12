"""Explicit casts involving GCC vector types (`vector_size` attribute).

A vector value is represented by its address (like a struct), so:
  * vector -> vector of the same size, and vector -> scalar, are handled in
    codegen (`expr_Cast`) as a bit reinterpretation of the same bytes;
  * scalar -> vector of the same size needs storage: it is lowered here to
    `(tmp_store, tmp)` where `tmp` is an anonymous local of the vector type
    and `tmp_store` writes the scalar bits into it through a casted pointer.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.typesys import TY_INT, pointer_to


def lower_scalar_to_vector(parser, loc, operand: N.Expr, ty) -> N.Expr | None:
    """Return the lowered node, or None when the cast needs no lowering."""
    if not ty.is_vector or operand.ty.is_vector:
        return None
    if operand.ty.size != ty.size:
        return None
    tmp = parser.new_temp(ty, loc)
    addr = parser.typed(N.AddrOf(loc, parser.typed(N.VarRef(loc, tmp))))
    ptr = parser.typed(N.Cast(loc, addr, ty=pointer_to(operand.ty)))
    store = parser.typed(N.Assign(loc, parser.typed(N.Deref(loc, ptr)), operand))
    return parser.typed(N.Comma(loc, store, parser.typed(N.VarRef(loc, tmp))))


def _vec_elem(parser, loc, var, index: int, elem_ty) -> N.Expr:
    """`var[index]` for a vector temp, via a pointer to its first element."""
    from hcc.sema.arith import new_add
    addr = parser.typed(N.AddrOf(loc, parser.typed(N.VarRef(loc, var))))
    ptr = parser.typed(N.Cast(loc, addr, ty=pointer_to(elem_ty)))
    idx = N.IntLit(loc, index, ty=TY_INT)
    return parser.typed(N.Deref(loc, parser.typed(new_add(ptr, idx, loc))))


def lower_convertvector(parser, loc, src: N.Expr, dst_ty) -> N.Expr:
    """__builtin_convertvector(src, dst_ty): element-wise conversion between two
    vectors with the same number of elements. Lowered to
    `(tmp_src = src, tmp_dst[0] = (E)tmp_src[0], ..., tmp_dst)`."""
    from hcc.sema.conversions import new_cast
    if not (src.ty.is_vector and dst_ty.is_vector):
        parser.error("__builtin_convertvector expects vector operands")
    n_src = src.ty.size // src.ty.base.size
    n_dst = dst_ty.size // dst_ty.base.size
    if n_src != n_dst:
        parser.error("__builtin_convertvector: vectors have different numbers of elements")
    tmp_src = parser.new_temp(src.ty, loc)
    tmp_dst = parser.new_temp(dst_ty, loc)
    node = parser.typed(N.VarRef(loc, tmp_dst))
    for i in reversed(range(n_dst)):
        elem = _vec_elem(parser, loc, tmp_src, i, src.ty.base)
        dst_elem = _vec_elem(parser, loc, tmp_dst, i, dst_ty.base)
        store = parser.typed(N.Assign(loc, dst_elem, new_cast(elem, dst_ty.base)))
        node = parser.typed(N.Comma(loc, store, node))
    first = parser.typed(N.Assign(loc, parser.typed(N.VarRef(loc, tmp_src)), src))
    return parser.typed(N.Comma(loc, first, node))


__all__ = ["lower_scalar_to_vector", "lower_convertvector"]
