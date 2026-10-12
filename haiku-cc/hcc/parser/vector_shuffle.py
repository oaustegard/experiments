"""__builtin_shuffle(v, mask) and __builtin_shuffle(v0, v1, mask) on GCC
vector types, lowered to plain element stores.

Result element i is `src[k & (n-1)]`, where k = (long)mask[i] and src is v0
or, in the two-operand form when `k & n` is set, v1. Lowered to:
    (tmp_v0 = v0, tmp_m = mask, tmp_r[0] = ..., ..., tmp_r)
Dynamic indices are read through a pointer cast to the element type (like
vector_broadcast.py), so the mask may be a run-time value.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.arith import new_add, new_binop
from hcc.sema.conversions import new_cast
from hcc.typesys import TY_LONG, pointer_to


def _elem(parser, loc, var, index, elem_ty) -> N.Expr:
    """Element `index` (an integer expression) of the vector stored in `var`."""
    addr = parser.typed(N.AddrOf(loc, parser.typed(N.VarRef(loc, var))))
    base = parser.typed(N.Cast(loc, addr, ty=pointer_to(elem_ty)))
    return parser.typed(N.Deref(loc, parser.typed(new_add(base, index, loc))))


def _long(loc, value: int) -> N.Expr:
    return N.IntLit(loc, value, ty=TY_LONG)


def lower_shuffle(parser, loc, args) -> N.Expr:
    """Return the lowered expression for __builtin_shuffle with `args`."""
    if len(args) not in (2, 3):
        parser.error("__builtin_shuffle expects 2 or 3 arguments")
    src_exprs, mask = args[:-1], args[-1]
    vec_ty = src_exprs[0].ty
    mask_ty = mask.ty
    if not vec_ty.is_vector or not mask_ty.is_vector:
        parser.error("__builtin_shuffle expects vector operands")
    if len(src_exprs) == 2 and not src_exprs[1].ty.is_vector:
        parser.error("__builtin_shuffle expects vector operands")
    elem = vec_ty.base
    count = vec_ty.size // elem.size
    if mask_ty.size // mask_ty.base.size != count:
        parser.error("__builtin_shuffle mask length must match the vector length")

    # Evaluate every operand exactly once, into anonymous locals.
    prologue = []
    temps = []
    for e in src_exprs + [mask]:
        tmp = parser.new_temp(e.ty, loc)
        prologue.append(parser.typed(N.Assign(loc, parser.typed(N.VarRef(loc, tmp)), e)))
        temps.append(tmp)
    src_tmps, mask_tmp = temps[:-1], temps[-1]
    res = parser.new_temp(vec_ty, loc)

    stores = []
    for i in range(count):
        k = parser.typed(new_cast(_elem(parser, loc, mask_tmp, _long(loc, i),
                                        mask_ty.base), TY_LONG))
        low = parser.typed(new_binop("&", k, _long(loc, count - 1), loc))
        value = parser.typed(_elem(parser, loc, src_tmps[0], low, elem))
        if len(src_tmps) == 2:
            high = parser.typed(new_binop("&", k, _long(loc, count), loc))
            other = parser.typed(_elem(parser, loc, src_tmps[1], low, elem))
            value = parser.typed(N.Cond(loc, high, other, value))
            value.ty = elem
        slot = parser.typed(_elem(parser, loc, res, _long(loc, i), elem))
        stores.append(parser.typed(N.Assign(loc, slot, value)))

    result = parser.typed(N.VarRef(loc, res))
    for store in reversed(stores):          # store_0 runs first
        result = parser.typed(N.Comma(loc, store, result))
    for assign in reversed(prologue):       # operands before the stores
        result = parser.typed(N.Comma(loc, assign, result))
    return result


__all__ = ["lower_shuffle"]
