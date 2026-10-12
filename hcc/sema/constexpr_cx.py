"""C23 constexpr objects: recognizes the initialization of a constexpr temporary.

Used by sema/constexpr.py: a `(constexpr T){...}` temp was checked constant when it
was declared, so a folding walk may skip its initializing stores.
"""
from __future__ import annotations

from hcc import nodes as N


def is_cx_init(node) -> bool:
    """Initialization of a C23 constexpr object (a constant, checked when it was
    declared), e.g. the temp of `(constexpr int){0}`: folding may skip it."""
    t = type(node)
    if t is N.Comma:
        return is_cx_init(node.lhs) and is_cx_init(node.rhs)
    if t is N.MemZero:
        return bool(getattr(node.var, "is_constexpr", False))
    if t is N.Assign:
        root = node.lhs                     # member/element of the constexpr object
        while type(root) in (N.Member_, N.Deref, N.Cast, N.Binary):
            root = root.operand if type(root) in (N.Member_, N.Deref, N.Cast) else root.lhs
        return type(root) is N.VarRef and bool(getattr(root.var, "is_constexpr", False))
    return False
