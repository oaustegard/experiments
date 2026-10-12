"""Operand check for prefix/postfix `++` and `--`: the operand must not be const-qualified."""

from __future__ import annotations

from hcc import nodes as N


def _is_const_lvalue(node) -> bool:
    """True if the lvalue `node` has a const-qualified type (a member of a const struct is const too)."""
    if node.ty is not None and node.ty.is_const:
        return True
    if isinstance(node, N.Member_):
        return _is_const_lvalue(node.operand)
    return False


def check_incdec_operand(parser, node, tok) -> None:
    """Reject `++`/`--` on a const object (`const int n; n++;`, or a const parameter)."""
    if _is_const_lvalue(parser.typed(node)):
        parser.error("increment of read-only location", tok)
