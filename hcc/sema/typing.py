"""add_type(node): assign `ty` to an expression node whose children are typed.

Called by the parser (Parser.typed) right after building each Expr.
Nodes that already have `ty` set (literals, Cast, lowered pointer arithmetic)
are left alone. Implicit conversions are inserted here as Cast nodes.
One function per node class, registered in _HANDLERS.
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc.sema.conversions import int_promote, new_cast, usual_arith_conv
from hcc.sema.constexpr import try_eval_const
from hcc.sema.void_value import require_value
from hcc.sema.incomplete import check_value_operands
from hcc.typesys import (TY_INT, TY_VOID, Type, TypeKind, is_decimal, is_dyn_struct, is_integer, is_numeric, pointer_to,
                         type_str)


def add_type(node) -> None:
    if node is None or getattr(node, "ty", None) is not None:
        return
    h = _HANDLERS.get(type(node))
    if h is None:
        diag.error(node, f"sema: cannot type {type(node).__name__}", "sema")
    check_value_operands(node)
    h(node)


def _rvalue(t: Type) -> Type:
    """Type of a non-lvalue result: lvalue conversion drops qualifiers of scalars (C11 6.3.2.1p2)."""
    if t is None or t.kind in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY, TypeKind.VLA,
                               TypeKind.VOID):
        return t
    return t.unqual()


def _var(n: N.VarRef):
    n.ty = n.var.ty


def _unary(n: N.Unary):
    require_value(n.operand)
    t = n.operand.ty
    if n.op == "!":
        n.ty = TY_INT
        return
    if n.op in ("~", "-") and t.is_vector and (is_integer(t.base) or
                                                (n.op == "-" and is_decimal(t.base))):
        n.ty = t                    # element-wise complement/negation keeps the vector type
        return
    if not is_numeric(t):
        diag.error(n, f"invalid operand to unary '{n.op}' (have '{t}')", "sema")
    if n.op == "~" and not is_integer(t):
        diag.error(n, "wrong type argument to bit-complement", "sema")
    n.operand = new_cast(n.operand, int_promote(t))
    n.ty = _rvalue(n.operand.ty)


def _addr(n: N.AddrOf):
    t = n.operand.ty
    n.ty = pointer_to(t)     # &array is a pointer to the whole array, not its element


def _deref(n: N.Deref):
    t = n.operand.ty
    if t.kind == TypeKind.FUNC:        # *f where f is a function designator
        n.ty = t
        return
    if t.base is None:
        diag.error(n, f"invalid type argument of unary '*' (have '{t}')", "sema")
    n.ty = t.base


SHIFTS = ("<<", ">>")
COMPARES = ("==", "!=", "<", "<=", ">", ">=")
ARITH_ONLY = ("*", "/", "%", "&", "|", "^")
NON_ARITH_KINDS = (TypeKind.PTR, TypeKind.ARRAY, TypeKind.VLA, TypeKind.FUNC,
                   TypeKind.STRUCT, TypeKind.UNION, TypeKind.VOID)


def _check_arith_operands(n: N.Binary):
    """`* / % & | ^` need arithmetic operands; pointers, aggregates and
    void are rejected rather than silently converted (C11 6.5.5-6.5.12)."""
    for t in (n.lhs.ty, n.rhs.ty):
        if not t.is_vector and t.kind in NON_ARITH_KINDS:
            diag.error(n, f"invalid operands to binary {n.op} ('{n.lhs.ty}' and '{n.rhs.ty}')", "sema")
            return


def _check_shift_operands(n: N.Binary):
    """Shifts need integer (or integer vector) operands; GCC rejects floats/pointers."""
    for side in (n.lhs, n.rhs):
        if side.ty.is_vector or is_integer(side.ty):
            continue
        diag.error(n.loc, f"invalid operands to binary {n.op} (have '{n.lhs.ty}' and '{n.rhs.ty}')", "sema")


def _binary(n: N.Binary):
    require_value(n.lhs)
    require_value(n.rhs)
    if n.op in ARITH_ONLY:
        _check_arith_operands(n)
    if n.op in SHIFTS:
        _check_shift_operands(n)
        n.lhs = new_cast(n.lhs, int_promote(n.lhs.ty))
        n.rhs = new_cast(n.rhs, int_promote(n.rhs.ty))
        n.ty = n.lhs.ty
        return
    if n.op in COMPARES:
        _check_nullptr_compare(n)
    n.lhs, n.rhs = usual_arith_conv(n.lhs, n.rhs)
    n.ty = TY_INT if n.op in COMPARES else n.lhs.ty


def _is_nullptr(node) -> bool:
    return isinstance(node, N.Cast) and node.nullptr_const


def _check_nullptr_compare(n):
    """C23 nullptr compares only with pointers or nullptr (ordered: never)."""
    l_np, r_np = _is_nullptr(n.lhs), _is_nullptr(n.rhs)
    if not (l_np or r_np):
        return
    if n.op not in ("==", "!="):
        diag.error(n, "ordered comparison with nullptr is invalid", "sema")
        return
    other = n.rhs if l_np else n.lhs
    if r_np and l_np:
        return
    if not (other.ty.kind == TypeKind.PTR or other.ty.kind == TypeKind.ARRAY
            or _is_nullptr(other) or _is_null_const(other)):
        diag.error(n, f"invalid operands to binary {n.op} ('{n.lhs.ty}' and '{n.rhs.ty}')", "sema")


def _logical(n: N.Logical):
    require_value(n.lhs)
    require_value(n.rhs)
    n.ty = TY_INT


def _assign(n: N.Assign):
    require_value(n.rhs)
    lt = n.lhs.ty
    if is_dyn_struct(lt) or is_dyn_struct(n.rhs.ty):
        diag.error(n, "unsupported: assignment of a struct with a variably modified member", "sema")
    if lt.kind == TypeKind.ARRAY and not lt.is_vector:
        diag.error(n, "assignment to expression with array type", "sema")
    _check_struct_scalar_mix(n, lt, _func_decay(n.rhs.ty))
    if lt.kind not in (TypeKind.STRUCT, TypeKind.UNION) and not lt.is_vector:
        n.rhs = new_cast(n.rhs, lt)
    n.ty = _rvalue(lt)      # the value of an assignment is an rvalue (qualifiers dropped)


def _check_struct_scalar_mix(n, lt, rt):
    """A struct/union never converts to or from a scalar (C11 6.5.16.1)."""
    if lt.is_vector or rt.is_vector:
        return
    l_agg = lt.kind in (TypeKind.STRUCT, TypeKind.UNION)
    r_agg = rt.kind in (TypeKind.STRUCT, TypeKind.UNION)
    l_scalar = lt.kind not in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY, TypeKind.VOID)
    r_scalar = rt.kind not in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY, TypeKind.VOID)
    if (l_agg and r_scalar) or (r_agg and l_scalar):
        diag.error(n, f"incompatible types when assigning to type '{type_str(lt)}' "
                      f"from type '{type_str(rt)}'", "sema")


def _is_null_const(node) -> bool:
    """True for an integer constant expression 0, or such an expression cast to `void *`
    (C11 6.3.2.3p3). A zero cast to another pointer type is not a null pointer constant."""
    if node.ty.kind == TypeKind.PTR:
        # only the cast spelling counts: a void * object that happens to fold to 0 does not
        if node.ty.base is None or node.ty.base.kind != TypeKind.VOID or not isinstance(node, N.Cast):
            return False
        if node.nullptr_const:
            return True
        node = node.operand
    elif not is_integer(node.ty):
        return False
    return try_eval_const(node) == 0


def _func_decay(t):
    """A function designator decays to a pointer to that function."""
    return pointer_to(t) if t.kind == TypeKind.FUNC else t


def _pointee_quals(t):
    """(const, volatile) of a pointee, looking through array element types (C23)."""
    c = v = False
    while True:
        c = c or t.is_const
        v = v or t.is_volatile
        if t.kind != TypeKind.ARRAY or t.base is None:
            return c, v
        t = t.base


def _void_ptr_composite(pa, pb):
    """Composite pointee when a `void *` meets another pointer: void carrying the
    union of both pointees' const/volatile; _Atomic is dropped."""
    c1, v1 = _pointee_quals(pa)
    c2, v2 = _pointee_quals(pb)
    t = TY_VOID.copy()
    t.is_const = c1 or c2
    t.is_volatile = v1 or v2
    t.is_atomic = False
    return pointer_to(t)


def _cond(n: N.Cond):
    require_value(n.cond)
    a, b = _func_decay(n.then.ty), _func_decay(n.els.ty)
    if a.kind == TypeKind.VOID or b.kind == TypeKind.VOID:
        # GCC extension (also accepted by gcc for pointer/scalar/struct partners):
        # if either operand is void, the result is void.
        n.ty = TY_VOID
    elif is_numeric(a) and is_numeric(b):
        n.then, n.els = usual_arith_conv(n.then, n.els)
        n.ty = _rvalue(n.then.ty)
    elif a.kind in (TypeKind.STRUCT, TypeKind.UNION):
        n.ty = a
    elif a.base is not None or b.base is not None:
        if a.base is not None and b.base is not None and _is_null_const(n.els):
            n.ty = a  # a null pointer constant takes the other operand's type
        elif a.base is not None and b.base is not None and _is_null_const(n.then):
            n.ty = b
        elif a.base is not None and b.base is not None and (
                a.base.kind == TypeKind.VOID or b.base.kind == TypeKind.VOID):
            n.ty = _void_ptr_composite(a.base, b.base)
        elif a.base is not None and b.base is not None and not a.base.is_complete \
                and b.base.is_complete:
            n.ty = b  # composite of `T (*)[]` and `T (*)[N]` is `T (*)[N]`
        else:
            n.ty = a if a.base is not None else b
    else:
        n.ty = a
    n.ty = _rvalue(n.ty)


def _comma(n: N.Comma):
    n.ty = _rvalue(n.rhs.ty)


def _member(n: N.Member_):
    n.ty = n.member.ty


def _stmt_expr(n: N.StmtExpr):
    items = [i for i in n.body.items if not isinstance(i, N.VLAExit)]
    n.ty = TY_VOID
    while items:
        last = items[-1]
        if isinstance(last, N.VLAExit):   # block-end VLA restore is not the value
            items = items[:-1]
            continue
        if isinstance(last, N.ExprStmt):
            n.ty = last.expr.ty
            return
        if isinstance(last, N.Label) and last.body is not None:
            items = [last.body]
            continue
        if isinstance(last, N.Block):
            items = last.items
            continue
        return


def _call(n: N.Call):
    # a call yields an rvalue: top-level qualifiers of a scalar return type are dropped (6.5.2.2p5)
    n.ty = _rvalue(n.func_ty.return_ty)


def _void(n):
    n.ty = TY_VOID


def _label_addr(n):
    n.ty = pointer_to(TY_VOID)


_HANDLERS = {
    N.VarRef: _var, N.Unary: _unary, N.AddrOf: _addr, N.Deref: _deref,
    N.Binary: _binary, N.Logical: _logical, N.Assign: _assign, N.Cond: _cond,
    N.Comma: _comma, N.Member_: _member, N.StmtExpr: _stmt_expr, N.Call: _call,
    N.MemZero: _void, N.LabelAddr: _label_addr,
}
