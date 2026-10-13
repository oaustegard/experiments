"""Evaluation order of `lhs op= rhs` (used by sema/arith.py compound_assign).

gcc evaluates the right operand, including any call in it, before it reads the
lhs value or computes the lhs address: `g += f()` sees the `g` that `f()` left
behind, and `a[idx()] += g()` calls g before idx. hcc's Binary is left-to-right,
so compound_assign parks such a rhs in a temp first. Side-effect-free rhs
expressions keep the plain lowering (their order is not observable)."""
from __future__ import annotations

from dataclasses import fields, is_dataclass

from hcc import nodes as N

# Fields that name declarations, types or locations rather than sub-expressions.
_SKIP_FIELDS = frozenset({"ty", "var", "member", "loc", "scope", "func_ty", "ret_buffer",
                          "result_buf", "type_arg", "label"})


def has_call(node) -> bool:
    """True if the expression tree under `node` contains a function call (or a
    setjmp-like call point), looking into statement expressions too."""
    if isinstance(node, (N.Call, N.StmtExpr)):
        return True
    if isinstance(node, N.BuiltinCall):
        return node.name == "setjmp" or any(has_call(a) for a in node.args)
    if isinstance(node, list):
        return any(has_call(x) for x in node)
    if not is_dataclass(node) or isinstance(node, type):
        return False
    return any(has_call(getattr(node, f.name, None)) for f in fields(node)
               if f.name not in _SKIP_FIELDS)
