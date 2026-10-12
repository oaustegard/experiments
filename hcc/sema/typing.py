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
from hcc.typesys import (TY_INT, TY_VOID, TypeKind, is_integer, is_numeric, pointer_to)


def add_type(node) -> None:
    if node is None or getattr(node, "ty", None) is not None:
        return
    h = _HANDLERS.get(type(node))
    if h is None:
        diag.error(node, f"sema: cannot type {type(node).__name__}", "sema")
    h(node)


def _var(n: N.VarRef):
    n.ty = n.var.ty


def _unary(n: N.Unary):
    t = n.operand.ty
    if n.op == "!":
        n.ty = TY_INT
        return
    if not is_numeric(t):
        diag.error(n, f"invalid operand to unary '{n.op}' (have '{t}')", "sema")
    if n.op == "~" and not is_integer(t):
        diag.error(n, "wrong type argument to bit-complement", "sema")
    n.operand = new_cast(n.operand, int_promote(t))
    n.ty = n.operand.ty


def _addr(n: N.AddrOf):
    t = n.operand.ty
    n.ty = pointer_to(t.base) if t.kind == TypeKind.ARRAY else pointer_to(t)


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


def _binary(n: N.Binary):
    if n.op in SHIFTS:
        n.lhs = new_cast(n.lhs, int_promote(n.lhs.ty))
        n.rhs = new_cast(n.rhs, int_promote(n.rhs.ty))
        n.ty = n.lhs.ty
        return
    n.lhs, n.rhs = usual_arith_conv(n.lhs, n.rhs)
    n.ty = TY_INT if n.op in COMPARES else n.lhs.ty


def _logical(n: N.Logical):
    n.ty = TY_INT


def _assign(n: N.Assign):
    lt = n.lhs.ty
    if lt.kind == TypeKind.ARRAY:
        diag.error(n, "assignment to expression with array type", "sema")
    if lt.kind not in (TypeKind.STRUCT, TypeKind.UNION):
        n.rhs = new_cast(n.rhs, lt)
    n.ty = lt


def _cond(n: N.Cond):
    a, b = n.then.ty, n.els.ty
    if a.kind == TypeKind.VOID or b.kind == TypeKind.VOID:
        n.ty = TY_VOID
    elif is_numeric(a) and is_numeric(b):
        n.then, n.els = usual_arith_conv(n.then, n.els)
        n.ty = n.then.ty
    elif a.kind in (TypeKind.STRUCT, TypeKind.UNION):
        n.ty = a
    elif a.base is not None or a.kind == TypeKind.FUNC:
        if b.base is not None and a.base is not None and a.base.kind == TypeKind.VOID:
            n.ty = pointer_to(a.base)
        elif b.base is not None and b.base.kind == TypeKind.VOID:
            n.ty = pointer_to(b.base)
        else:
            n.ty = pointer_to(a.base) if a.base is not None else pointer_to(a)
    elif b.base is not None or b.kind == TypeKind.FUNC:
        n.ty = pointer_to(b.base) if b.base is not None else pointer_to(b)
    else:
        n.ty = a


def _comma(n: N.Comma):
    n.ty = n.rhs.ty


def _member(n: N.Member_):
    n.ty = n.member.ty


def _stmt_expr(n: N.StmtExpr):
    items = n.body.items
    n.ty = TY_VOID
    while items:
        last = items[-1]
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
    n.ty = n.func_ty.return_ty


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
