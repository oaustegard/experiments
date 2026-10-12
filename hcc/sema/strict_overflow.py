"""Overflow-undefined (signed) algebraic simplification (CONTRACT section 5).

Signed integer overflow is undefined behaviour in C (gcc's default; -fno-wrapv).
Because of that, `(x * c) / c` with a non-zero integer constant `c` may be
replaced by `x`: when the multiplication does not overflow the quotient is
exactly `x`, and when it does overflow the program has no defined meaning.
Unsigned types wrap (defined), so they are never rewritten.

    simplify_mul_div(node, loc)   node: typed N.Binary "/" -> replacement or node
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.constexpr import try_eval_const
from hcc.sema.conversions import new_cast
from hcc.typesys import is_integer


def _signed_int(ty) -> bool:
    return ty is not None and is_integer(ty) and not ty.is_unsigned


def simplify_mul_div(node, loc=None):
    """Return `x` for a typed `(x * c) / c` (signed, c a non-zero constant),
    otherwise return `node` unchanged."""
    if not isinstance(node, N.Binary) or node.op != "/":
        return node
    mul = node.lhs
    if not (isinstance(mul, N.Binary) and mul.op == "*"):
        return node
    if not (_signed_int(mul.ty) and _signed_int(node.ty) and mul.ty is node.ty):
        return node
    if try_eval_const(mul) is not None:
        # a constant product is folded with wrap-around semantics (as gcc does);
        # the UB rewrite only concerns products that depend on a runtime value
        return node
    divisor = try_eval_const(node.rhs)
    if divisor is None or divisor == 0:
        return node
    for const_side, other in ((mul.rhs, mul.lhs), (mul.lhs, mul.rhs)):
        if try_eval_const(const_side) == divisor and other.ty is mul.ty:
            return new_cast(other, node.ty)
    return node
