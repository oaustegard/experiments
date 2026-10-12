"""Operand check for unary '&': C6.5.3.2 requires an lvalue (or a function).

`is_rvalue_operand(node)` is True only for expression forms that are certainly
not lvalues (casts, calls, ?:, assignments, non-lvalue comma expressions, and
member access on any of these). Anything else is accepted, so this check never
rejects an lvalue form it does not know about.
"""
from __future__ import annotations

from hcc import nodes as N


def is_rvalue_operand(node) -> bool:
    if isinstance(node, (N.Cast, N.Call, N.Cond, N.Assign)):
        return True
    if isinstance(node, N.Comma):
        return not node.lvalue
    if isinstance(node, N.Member_):
        # a->m is Member_(Deref(a)): always an lvalue. s.m is an lvalue iff s is.
        return is_rvalue_operand(node.operand)
    return False
