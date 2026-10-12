"""Operand typing for __builtin_{add,sub,mul}_overflow[_p] when an operand or the
result is __int128 (the 64-bit forms are typed in builtins.py).

wide_overflow_operands(parser, name, args, res_ty) -> list[Expr] | None
    None: every operand and the result are at most 64 bits (narrow path).
    Otherwise both operands are converted to one 128-bit domain:
      * UINT128 when all operands are unsigned and one of them is 128-bit,
      * INT128 otherwise (narrow unsigned operands are zero-extended into it).
    Codegen (codegen/overflow128.py) then checks the exact result against res_ty.
Unsupported combinations are reported as errors instead of approximated.
"""
from __future__ import annotations

from hcc.sema.conversions import new_cast
from hcc.sema.constexpr import try_eval_const
from hcc import typesys as T


def _is_i128(ty) -> bool:
    return ty.kind == T.TypeKind.INT128


def _unsigned_operand(a) -> bool:
    """An operand counts as unsigned if its type is, or if it is a narrow signed
    constant that is not negative (exact in an unsigned 128-bit domain)."""
    if a.ty.is_unsigned:
        return True
    if a.ty.size >= 16:
        return False
    v = try_eval_const(a)
    return v is not None and v >= 0


def wide_overflow_operands(parser, name: str, args, res_ty):
    if not (any(_is_i128(a.ty) for a in args) or _is_i128(res_ty)):
        return None
    if name == "mul":
        parser.error("__builtin_mul_overflow with __int128 operands is not supported")
    all_unsigned = all(_unsigned_operand(a) for a in args)
    if all_unsigned and any(_is_i128(a.ty) for a in args):
        dom = T.TY_UINT128
        if not res_ty.is_unsigned and res_ty.kind != T.TypeKind.BOOL:
            parser.error("__builtin_%s_overflow: signed __int128 result with unsigned "
                         "__int128 operands is not supported" % name)
    else:
        if any(_is_i128(a.ty) and a.ty.is_unsigned for a in args):   # unsigned 128-bit + signed
            parser.error("__builtin_%s_overflow: mixing unsigned __int128 with a signed "
                         "operand is not supported" % name)
        dom = T.TY_INT128
        if res_ty.kind == T.TypeKind.INT128 and res_ty.is_unsigned and \
                any(_is_i128(a.ty) for a in args):
            parser.error("__builtin_%s_overflow: signed __int128 operands with an unsigned "
                         "__int128 result are not supported" % name)
    return [new_cast(a, dom) for a in args]
