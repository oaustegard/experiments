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
from hcc.typesys import pointer_to


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


__all__ = ["lower_scalar_to_vector"]
