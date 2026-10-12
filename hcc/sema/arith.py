"""Pointer arithmetic and assignment-operator lowering (CONTRACT section 5).

    new_add(lhs, rhs, loc)            num+num, ptr+int, int+ptr
    new_sub(lhs, rhs, loc)            num-num, ptr-int, ptr-ptr
    new_binop(op, lhs, rhs, loc)      dispatches + and - to the above
    compound_assign(op, lhs, rhs, loc, new_temp)    a op= b
    inc_dec(lhs, delta, is_post, loc, new_temp)     ++a, a++, --a, a--
`new_temp(ty)` is Parser.new_temp: creates an anonymous local.
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc.sema.conversions import new_cast
from hcc.sema.typing import add_type
from hcc.typesys import TY_INT, TY_LONG, TypeKind, is_numeric, pointer_to


def _t(node):
    add_type(node)
    return node


def _elem_size(ty) -> int:
    base = ty.base
    if base.kind in (TypeKind.VOID, TypeKind.FUNC):
        return 1
    return base.size


def _is_ptr(ty) -> bool:
    return ty.base is not None and ty.kind in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.VLA)


def new_add(lhs, rhs, loc):
    add_type(lhs)
    add_type(rhs)
    if is_numeric(lhs.ty) and is_numeric(rhs.ty):
        return _t(N.Binary(loc, "+", lhs, rhs))
    if _is_ptr(lhs.ty) and _is_ptr(rhs.ty):
        diag.error(loc, "invalid operands to binary +", "sema")
    if not _is_ptr(lhs.ty):
        lhs, rhs = rhs, lhs
    if not _is_ptr(lhs.ty):
        diag.error(loc, f"invalid operands to binary + ('{lhs.ty}' and '{rhs.ty}')", "sema")
    size = N.IntLit(loc, _elem_size(lhs.ty), ty=TY_LONG)
    off = N.Binary(loc, "*", new_cast(rhs, TY_LONG), size, ty=TY_LONG)
    return N.Binary(loc, "+", lhs, off, ty=pointer_to(lhs.ty.base))


def new_sub(lhs, rhs, loc):
    add_type(lhs)
    add_type(rhs)
    if is_numeric(lhs.ty) and is_numeric(rhs.ty):
        return _t(N.Binary(loc, "-", lhs, rhs))
    if _is_ptr(lhs.ty) and is_numeric(rhs.ty):
        size = N.IntLit(loc, _elem_size(lhs.ty), ty=TY_LONG)
        off = N.Binary(loc, "*", new_cast(rhs, TY_LONG), size, ty=TY_LONG)
        return N.Binary(loc, "-", lhs, off, ty=pointer_to(lhs.ty.base))
    if _is_ptr(lhs.ty) and _is_ptr(rhs.ty):
        diff = N.Binary(loc, "-", new_cast(lhs, TY_LONG), new_cast(rhs, TY_LONG), ty=TY_LONG)
        return N.Binary(loc, "/", diff, N.IntLit(loc, _elem_size(lhs.ty), ty=TY_LONG), ty=TY_LONG)
    diag.error(loc, f"invalid operands to binary - ('{lhs.ty}' and '{rhs.ty}')", "sema")


def new_binop(op, lhs, rhs, loc):
    if op == "+":
        return new_add(lhs, rhs, loc)
    if op == "-":
        return new_sub(lhs, rhs, loc)
    return _t(N.Binary(loc, op, lhs, rhs))


def compound_assign(op, lhs, rhs, loc, new_temp):
    """Lower `lhs op= rhs` so that lhs is evaluated only once."""
    add_type(lhs)
    add_type(rhs)
    if isinstance(lhs, N.VarRef):           # simple variable: no temp needed
        return _t(N.Assign(loc, lhs, new_binop(op, lhs, rhs, loc)))
    if isinstance(lhs, N.Member_) and lhs.member.is_bitfield:
        tmp = new_temp(pointer_to(lhs.operand.ty))
        e1 = _t(N.Assign(loc, _t(N.VarRef(loc, tmp)), _t(N.AddrOf(loc, lhs.operand))))

        def obj():
            return _t(N.Member_(loc, _t(N.Deref(loc, _t(N.VarRef(loc, tmp)))), lhs.member))
        e2 = _t(N.Assign(loc, obj(), new_binop(op, obj(), rhs, loc)))
        return _t(N.Comma(loc, e1, e2))
    tmp = new_temp(pointer_to(lhs.ty))
    e1 = _t(N.Assign(loc, _t(N.VarRef(loc, tmp)), _t(N.AddrOf(loc, lhs))))

    def deref():
        return _t(N.Deref(loc, _t(N.VarRef(loc, tmp))))
    e2 = _t(N.Assign(loc, deref(), new_binop(op, deref(), rhs, loc)))
    return _t(N.Comma(loc, e1, e2))


def inc_dec(lhs, delta: int, is_post: bool, loc, new_temp):
    add_type(lhs)
    one = N.IntLit(loc, delta, ty=TY_INT)
    if not is_post:
        return compound_assign("+", lhs, one, loc, new_temp)
    assign = compound_assign("+", lhs, one, loc, new_temp)
    back = new_add(assign, N.IntLit(loc, -delta, ty=TY_INT), loc)
    return new_cast(back, lhs.ty)
