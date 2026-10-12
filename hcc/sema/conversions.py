"""Integer promotion, usual arithmetic conversions and cast construction.

    int_promote(ty) -> Type
    common_type(a, b) -> Type           (C11 6.3.1.8, pointers win)
    new_cast(expr, ty) -> Expr          (typed Cast node; no-op casts elided)
    usual_arith_conv(lhs, rhs) -> (lhs', rhs')
    default_arg_promote(expr) -> Expr   (float -> double for varargs)
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc.typesys import (TY_DOUBLE, TY_INT, TY_LDOUBLE, TY_UINT, Type, TypeKind, is_flonum,
                       is_integer, pointer_to)

_FLOAT_RANK = {TypeKind.FLOAT16: 1, TypeKind.FLOAT: 2, TypeKind.DOUBLE: 3, TypeKind.LDOUBLE: 4,
               TypeKind.FLOAT128: 5}


def int_promote(ty: Type) -> Type:
    if ty.kind == TypeKind.ENUM:
        return TY_UINT if ty.is_unsigned else TY_INT
    if is_integer(ty) and ty.size < 4:
        return TY_INT
    return ty


def common_type(a: Type, b: Type) -> Type:
    if a.base is not None and a.kind in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.VLA):
        return pointer_to(a.base)
    if a.kind == TypeKind.FUNC:
        return pointer_to(a)
    if b.kind == TypeKind.FUNC:
        return pointer_to(b)
    if is_flonum(a) or is_flonum(b):
        ra = _FLOAT_RANK.get(a.kind, 0)
        rb = _FLOAT_RANK.get(b.kind, 0)
        return a if ra >= rb else b
    a = int_promote(a)
    b = int_promote(b)
    if a.size != b.size:
        return a if a.size > b.size else b
    if b.is_unsigned:
        return b
    return a


def same_scalar(a: Type, b: Type) -> bool:
    return a.kind == b.kind and a.size == b.size and a.is_unsigned == b.is_unsigned and \
        a.kind not in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY)


def new_cast(expr: N.Expr, ty: Type) -> N.Expr:
    from hcc.sema.typing import add_type
    add_type(expr)
    if expr.ty is ty:
        return expr
    if expr.ty.is_complex or ty.is_complex:
        # complex conversions are lowered by the parser (parser/complex.py); reaching here is a gap
        diag.error(expr, f"unsupported conversion from '{expr.ty}' to '{ty}'", "sema")
    if same_scalar(expr.ty, ty) and ty.kind != TypeKind.PTR:
        return expr
    c = N.Cast(expr.loc, expr, ty=ty.unqual() if ty.kind != TypeKind.VOID else ty)
    return c


def _bitfield_operand_type(expr: N.Expr) -> Type:
    """C11 6.3.1.1: a bit-field narrower than int promotes to int, whatever
    its declared type (`long x:24` is an int operand)."""
    m = getattr(expr, "member", None)
    if type(expr) is N.Member_ and m is not None and m.is_bitfield and \
            is_integer(expr.ty) and m.bit_width < 32:
        return TY_INT
    return expr.ty


def usual_arith_conv(lhs: N.Expr, rhs: N.Expr):
    ty = common_type(_bitfield_operand_type(lhs), _bitfield_operand_type(rhs))
    return new_cast(lhs, ty), new_cast(rhs, ty)


def default_arg_promote(expr: N.Expr) -> N.Expr:
    # _FloatN types are not subject to default argument promotions (TS 18661-3).
    if expr.ty.kind == TypeKind.FLOAT and not expr.ty.is_floatn:
        return new_cast(expr, TY_DOUBLE)
    if is_integer(expr.ty) and expr.ty.size < 4:
        return new_cast(expr, int_promote(expr.ty))
    return expr


__all__ = ["int_promote", "common_type", "new_cast", "usual_arith_conv",
           "default_arg_promote", "TY_LDOUBLE"]
