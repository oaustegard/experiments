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
from hcc.sema.bitfield_width import narrow_wide_bitfield_result
from hcc.sema.compat import is_compatible
from hcc.sema.conversions import new_cast
from hcc.sema.typing import add_type
from hcc.sema.void_value import require_value
from hcc.typesys import (TY_INT, TY_LONG, TypeKind, int_type_for, is_decimal, is_flonum, is_integer,
                         is_numeric, pointer_to, vector_of)


def _t(node):
    add_type(node)
    return node


def _elem_size(ty, loc):
    """Byte size of one pointee/element as an integer expression. Pointees
    that are VLAs (bound to a hidden length local) give a run-time product."""
    base = ty.base
    if base.kind in (TypeKind.VOID, TypeKind.FUNC):
        return N.IntLit(loc, 1, ty=TY_LONG)
    return _size_expr(base, loc)


def _size_expr(t, loc):
    if t.kind == TypeKind.VLA and t.vla_len is not None:
        n = new_cast(t.vla_len, TY_LONG)
        return _t(N.Binary(loc, "*", n, _size_expr(t.base, loc), ty=TY_LONG))
    return N.IntLit(loc, t.size, ty=TY_LONG)


def _func_decay(node):
    """A function designator used as an arithmetic operand decays to a pointer."""
    if node.ty.kind == TypeKind.FUNC:
        return new_cast(node, pointer_to(node.ty))
    return node


def _is_ptr(ty) -> bool:
    return ty.base is not None and ty.kind in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.VLA)


def new_add(lhs, rhs, loc):
    add_type(lhs)
    add_type(rhs)
    require_value(lhs)
    require_value(rhs)
    lhs, rhs = _func_decay(lhs), _func_decay(rhs)
    if is_numeric(lhs.ty) and is_numeric(rhs.ty):
        return _t(narrow_wide_bitfield_result("+", lhs, rhs, _t(N.Binary(loc, "+", lhs, rhs))))
    if _is_ptr(lhs.ty) and _is_ptr(rhs.ty):
        diag.error(loc, "invalid operands to binary +", "sema")
    if not _is_ptr(lhs.ty):
        lhs, rhs = rhs, lhs
    if not _is_ptr(lhs.ty):
        diag.error(loc, f"invalid operands to binary + ('{lhs.ty}' and '{rhs.ty}')", "sema")
    size = _elem_size(lhs.ty, loc)
    off = N.Binary(loc, "*", new_cast(rhs, TY_LONG), size, ty=TY_LONG)
    return N.Binary(loc, "+", lhs, off, ty=pointer_to(lhs.ty.base))


def _pointees_compatible(a, b) -> bool:
    """C11 6.5.6p3: pointer difference needs pointers to compatible object types,
    ignoring const/volatile/restrict on the pointees (_Atomic is not ignored)."""
    def strip(t):
        if not (t.is_const or t.is_volatile or t.is_restrict):
            return t
        u = t.copy()
        u.is_const = u.is_volatile = u.is_restrict = False
        return u
    return is_compatible(strip(a), strip(b))


def new_sub(lhs, rhs, loc):
    add_type(lhs)
    add_type(rhs)
    require_value(lhs)
    require_value(rhs)
    lhs, rhs = _func_decay(lhs), _func_decay(rhs)
    if is_numeric(lhs.ty) and is_numeric(rhs.ty):
        return _t(narrow_wide_bitfield_result("-", lhs, rhs, _t(N.Binary(loc, "-", lhs, rhs))))
    if _is_ptr(lhs.ty) and is_numeric(rhs.ty):
        size = _elem_size(lhs.ty, loc)
        off = N.Binary(loc, "*", new_cast(rhs, TY_LONG), size, ty=TY_LONG)
        return N.Binary(loc, "-", lhs, off, ty=pointer_to(lhs.ty.base))
    if _is_ptr(lhs.ty) and _is_ptr(rhs.ty):
        if not _pointees_compatible(lhs.ty.base, rhs.ty.base):
            diag.error(loc, f"invalid operands to binary - ('{lhs.ty}' and '{rhs.ty}')", "sema")
        diff = N.Binary(loc, "-", new_cast(lhs, TY_LONG), new_cast(rhs, TY_LONG), ty=TY_LONG)
        return N.Binary(loc, "/", diff, _elem_size(lhs.ty, loc), ty=TY_LONG)
    diag.error(loc, f"invalid operands to binary - ('{lhs.ty}' and '{rhs.ty}')", "sema")


VECTOR_OPS = ("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>")
VECTOR_CMP_OPS = ("==", "!=", "<", "<=", ">", ">=")


def vector_binop(op, lhs, rhs, loc):
    """Element-wise `lhs op rhs` on two GCC vectors of equal size (integer
    elements for all VECTOR_OPS, floating elements for + - * /).
    Comparisons on integer vectors yield a signed-integer vector of the same
    width whose lanes are all-ones (true) or zero (false).
    Returns None when the operands are not such a pair."""
    if not (op in VECTOR_OPS or op in VECTOR_CMP_OPS):
        return None
    if not (lhs.ty.is_vector and rhs.ty.is_vector):
        return None
    if lhs.ty.size != rhs.ty.size:
        return None
    if op in VECTOR_CMP_OPS:
        # float/double lanes compare too (ordered predicates, GCC semantics)
        if not (is_integer(lhs.ty.base) or (is_flonum(lhs.ty.base) and lhs.ty.base.size in (4, 8))):
            return None
        res = vector_of(int_type_for(lhs.ty.base.size, False), lhs.ty.size)  # lanes: -1 / 0
        return _t(N.Binary(loc, op, lhs, rhs, ty=res))
    if is_flonum(lhs.ty.base) or is_decimal(lhs.ty.base):
        if op not in ("+", "-", "*", "/"):
            return None
    elif not is_integer(lhs.ty.base):
        return None
    return _t(N.Binary(loc, op, lhs, rhs, ty=lhs.ty))


def new_binop(op, lhs, rhs, loc):
    add_type(lhs)
    add_type(rhs)
    require_value(lhs)
    require_value(rhs)
    vec = vector_binop(op, lhs, rhs, loc)
    if vec is not None:
        return vec
    if (op in VECTOR_OPS or op in VECTOR_CMP_OPS) and (lhs.ty.is_vector or rhs.ty.is_vector):
        diag.error(loc, f"invalid operands to binary {op} ('{lhs.ty}' and '{rhs.ty}')", "sema")
    if op == "+":
        return new_add(lhs, rhs, loc)
    if op == "-":
        return new_sub(lhs, rhs, loc)
    node = _t(N.Binary(loc, op, lhs, rhs))
    return _t(narrow_wide_bitfield_result(op, lhs, rhs, node))


def _binop_buf(op, lhs, rhs, loc, new_temp):
    """new_binop, plus the result buffer a vector-valued Binary needs."""
    node = new_binop(op, lhs, rhs, loc)
    if isinstance(node, N.Binary) and node.ty.is_vector and node.result_buf is None:
        node.result_buf = new_temp(node.ty)
    return node


def compound_assign(op, lhs, rhs, loc, new_temp):
    """Lower `lhs op= rhs` so that lhs is evaluated only once."""
    add_type(lhs)
    add_type(rhs)
    require_value(lhs)
    require_value(rhs)
    if isinstance(lhs, N.VarRef):           # simple variable: no temp needed
        return _t(N.Assign(loc, lhs, _binop_buf(op, lhs, rhs, loc, new_temp)))
    if isinstance(lhs, N.Member_) and lhs.member.is_bitfield:
        tmp = new_temp(pointer_to(lhs.operand.ty))
        e1 = _t(N.Assign(loc, _t(N.VarRef(loc, tmp)), _t(N.AddrOf(loc, lhs.operand))))

        def obj():
            return _t(N.Member_(loc, _t(N.Deref(loc, _t(N.VarRef(loc, tmp)))), lhs.member))
        e2 = _t(N.Assign(loc, obj(), _binop_buf(op, obj(), rhs, loc, new_temp)))
        return _t(N.Comma(loc, e1, e2))
    tmp = new_temp(pointer_to(lhs.ty))
    e1 = _t(N.Assign(loc, _t(N.VarRef(loc, tmp)), _t(N.AddrOf(loc, lhs))))

    def deref():
        return _t(N.Deref(loc, _t(N.VarRef(loc, tmp))))
    e2 = _t(N.Assign(loc, deref(), _binop_buf(op, deref(), rhs, loc, new_temp)))
    return _t(N.Comma(loc, e1, e2))


def _addressed_place(lhs, loc, new_temp):
    """Return (pre, make_place): `pre` evaluates the address of lhs once into
    a temp; make_place() rebuilds an lvalue for that same object. Bitfields
    have no address, so their container's address is taken instead."""
    if isinstance(lhs, N.Member_) and lhs.member.is_bitfield:
        tmp = new_temp(pointer_to(lhs.operand.ty))
        pre = [_t(N.Assign(loc, _t(N.VarRef(loc, tmp)), _t(N.AddrOf(loc, lhs.operand))))]
        member = lhs.member

        def make_place():
            return _t(N.Member_(loc, _t(N.Deref(loc, _t(N.VarRef(loc, tmp)))), member))
        return pre, make_place
    tmp = new_temp(pointer_to(lhs.ty))
    pre = [_t(N.Assign(loc, _t(N.VarRef(loc, tmp)), _t(N.AddrOf(loc, lhs))))]

    def make_place():
        return _t(N.Deref(loc, _t(N.VarRef(loc, tmp))))
    return pre, make_place


def inc_dec(lhs, delta: int, is_post: bool, loc, new_temp):
    """++/--. Post forms yield the old value, saved in a temp before the
    update (reversing the arithmetic is wrong when the store truncates)."""
    add_type(lhs)
    one = N.IntLit(loc, delta, ty=TY_INT)
    if not is_post:
        return compound_assign("+", lhs, one, loc, new_temp)
    if isinstance(lhs, N.VarRef):
        pre, make_place = [], (lambda: lhs)
    else:
        pre, make_place = _addressed_place(lhs, loc, new_temp)
    old = new_temp(lhs.ty)
    save = _t(N.Assign(loc, _t(N.VarRef(loc, old)), make_place()))
    update = compound_assign("+", make_place(), one, loc, new_temp)
    result = _t(N.Comma(loc, save, _t(N.Comma(loc, update, _t(N.VarRef(loc, old))))))
    for stmt in reversed(pre):
        result = _t(N.Comma(loc, stmt, result))
    return result
