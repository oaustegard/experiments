"""Bit-field width expressions (C11 6.7.2.1p4): must be an integer constant expression.

GCC rejects pointer/float widths and any width that names a variable, even a
const-qualified one (the block-scope `const int` folding GNU applies elsewhere
is not an integer constant expression here).
"""
from __future__ import annotations

import dataclasses

from hcc import diag
from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.constexpr import eval_const


def _names_variable(node) -> bool:
    stack = [node]
    while stack:
        n = stack.pop()
        # C23: a constexpr object is itself a constant (GCC accepts it as a width)
        if type(n) is N.VarRef and not getattr(n.var, "is_constexpr", False):
            return True
        if isinstance(n, N.Node) and dataclasses.is_dataclass(n):
            for f in dataclasses.fields(n):
                v = getattr(n, f.name, None)
                if isinstance(v, N.Node):
                    stack.append(v)
                elif isinstance(v, (list, tuple)):
                    stack.extend(x for x in v if isinstance(x, N.Node))
    return False


def bitfield_width_value(node, name, where) -> int:
    """Value of a bit-field width expression, diagnosing non-constant ones."""
    ty = getattr(node, "ty", None)
    if ty is None or not T.is_integer(ty) or _names_variable(node):
        what = "bit-field" if name is None else f"bit-field '{name.text}'"
        diag.error(name or where, f"{what} width not an integer constant")
    return eval_const(node)
