"""Operand check for prefix/postfix `++` and `--`: the operand must be a modifiable lvalue."""

from __future__ import annotations

from hcc import nodes as N


def _is_const_lvalue(node) -> bool:
    """True if the lvalue `node` has a const-qualified type (a member of a const struct is const too)."""
    if node.ty is not None and node.ty.is_const:
        return True
    if isinstance(node, N.Member_):
        return _is_const_lvalue(node.operand)
    return False


def _is_lvalue_form(node) -> bool:
    """True for expression forms that designate an object (C11 6.3p3)."""
    if isinstance(node, (N.VarRef, N.Deref)):
        return True
    if isinstance(node, N.Comma):
        return bool(node.lvalue)           # compound literal
    if isinstance(node, N.Member_):
        return _is_lvalue_form(node.operand)
    return False


def check_incdec_operand(parser, node, tok) -> None:
    """Reject `++`/`--` on a non-lvalue (`7++`) or on a const object (`const int n; n++;`)."""
    if not _is_lvalue_form(node):
        parser.error("lvalue required as increment operand", tok)
    if _is_const_lvalue(parser.typed(node)):
        parser.error("increment of read-only location", tok)
