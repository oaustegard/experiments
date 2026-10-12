"""Constant folding of comparisons whose truth values are provably equal.

For signed integers, overflow is undefined, so `a + p REL b + q` is the same
constraint as `a - b REL' (q - p)` (strict relations become non-strict on
integers). Each comparison is normalised to a canonical key over the two
variables; comparing two such comparisons with `==` or `!=` then folds to a
constant when their keys match.

    fold_rel_equal(node, const) -> 0 | 1 | None   (node: Binary ==/!=)
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.typesys import TypeKind, is_integer

_RELS = ("<", "<=", ">", ">=", "==", "!=")
_MIRROR = {"<=": ">=", ">=": "<=", "==": "==", "!=": "!="}


def _signed_int(ty) -> bool:
    return ty is not None and is_integer(ty) and not ty.is_unsigned and ty.kind != TypeKind.PTR


def _lin(node, const):
    """(var, k) when `node` is `var + k` for a plain signed integer variable."""
    t = type(node)
    if t is N.VarRef:
        var = node.var
        if var.ty.is_volatile or var.ty.kind in (TypeKind.ARRAY, TypeKind.FUNC):
            return None
        return (var, 0) if _signed_int(node.ty) else None
    if t is N.Binary and node.op in ("+", "-") and _signed_int(node.ty):
        left = _lin(node.lhs, const)
        if left is not None:
            c = const(node.rhs)
            if c is None:
                return None
            return left[0], left[1] + (c if node.op == "+" else -c)
        if node.op == "+":
            right = _lin(node.rhs, const)
            c = const(node.lhs)
            if right is not None and c is not None:
                return right[0], right[1] + c
    return None


def _key(cmp, const):
    """Canonical (id(a), id(b), op, c) for the constraint `a - b op c`, where
    op is one of <=, >=, ==, != and a has the smaller id; None if not of that form."""
    if type(cmp) is not N.Binary or cmp.op not in _RELS:
        return None
    lhs, rhs = _lin(cmp.lhs, const), _lin(cmp.rhs, const)
    if lhs is None or rhs is None or lhs[0] is rhs[0]:
        return None
    (a, p), (b, q) = lhs, rhs
    op, c = cmp.op, q - p
    if op == "<":
        op, c = "<=", c - 1
    elif op == ">":
        op, c = ">=", c + 1
    if id(a) > id(b):
        a, b, op, c = b, a, _MIRROR[op], -c
    return id(a), id(b), op, c


def fold_rel_equal(node, const):
    """Value of `X == Y` / `X != Y` when both are comparisons with the same
    constraint (e.g. `(a < b) != (b > a)`), else None."""
    kx = _key(node.lhs, const)
    if kx is None or kx != _key(node.rhs, const):
        return None
    return int(node.op == "==")
