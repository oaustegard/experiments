"""__builtin_shufflevector(v1, v2, i0, i1, ...) on GCC vector types, lowered to
plain element stores (like vector_shuffle.py, but the indices are constants).

Index k selects element k of the concatenation v1:v2 (k < len(v1) picks from
v1, otherwise element k - len(v1) of v2); -1 means "don't care" and is read as
element 0. The result has one element per index and the operands' element type.
Lowered to:  (tmp1 = v1, tmp2 = v2, res[0] = ..., ..., res)
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.arith import new_add
from hcc.sema.compat import is_compatible
from hcc.sema.constexpr import try_eval_const
from hcc.typesys import TY_LONG, pointer_to, vector_of


def _elem(parser, loc, var, index: int, elem_ty) -> N.Expr:
    """Element `index` (constant) of the vector stored in `var`."""
    addr = parser.typed(N.AddrOf(loc, parser.typed(N.VarRef(loc, var))))
    base = parser.typed(N.Cast(loc, addr, ty=pointer_to(elem_ty)))
    off = N.IntLit(loc, index, ty=TY_LONG)
    return parser.typed(N.Deref(loc, parser.typed(new_add(base, off, loc))))


def lower_shufflevector(parser, loc, args) -> N.Expr:
    """Return the lowered expression for __builtin_shufflevector(args...)."""
    if len(args) < 3:
        parser.error("__builtin_shufflevector expects at least 3 arguments")
    v1, v2 = args[0], args[1]
    if not v1.ty.is_vector or not v2.ty.is_vector:
        parser.error("__builtin_shufflevector expects vector operands")
    elem = v1.ty.base
    if not is_compatible(elem, v2.ty.base):
        parser.error("__builtin_shufflevector vector operands must have the same element type")
    n1 = v1.ty.size // elem.size
    n2 = v2.ty.size // elem.size
    indices = []
    for a in args[2:]:
        k = try_eval_const(a)
        if k is None:
            parser.error("__builtin_shufflevector index must be an integer constant")
        if k != -1 and not 0 <= k < n1 + n2:
            parser.error("__builtin_shufflevector index is out of range")
        indices.append(0 if k == -1 else k)
    res_ty = vector_of(elem, len(indices) * elem.size)

    # Evaluate each operand exactly once, into anonymous locals.
    t1 = parser.new_temp(v1.ty, loc)
    t2 = parser.new_temp(v2.ty, loc)
    prologue = [parser.typed(N.Assign(loc, parser.typed(N.VarRef(loc, t1)), v1)),
                parser.typed(N.Assign(loc, parser.typed(N.VarRef(loc, t2)), v2))]
    res = parser.new_temp(res_ty, loc)

    stores = []
    for i, k in enumerate(indices):
        src, j = (t1, k) if k < n1 else (t2, k - n1)
        value = _elem(parser, loc, src, j, elem)
        slot = _elem(parser, loc, res, i, elem)
        stores.append(parser.typed(N.Assign(loc, slot, value)))

    result = parser.typed(N.VarRef(loc, res))
    for store in reversed(stores):          # store_0 runs first
        result = parser.typed(N.Comma(loc, store, result))
    for assign in reversed(prologue):       # operands before the stores
        result = parser.typed(N.Comma(loc, assign, result))
    return result


__all__ = ["lower_shufflevector"]
