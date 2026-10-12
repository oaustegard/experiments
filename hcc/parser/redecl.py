"""Ordinary-identifier redeclaration checks (C11 6.7p3, 6.2.1p4).

Within one scope an ordinary identifier names exactly one thing: an object,
a function, a typedef or an enumeration constant. Only a few combinations are
legal: a typedef repeated as the same typedef, a function repeated as a
function, and (at file scope) tentative object definitions. Block-scope
objects may not repeat a block-scope object of the same scope.
"""
from __future__ import annotations

from hcc import nodes as N


def _old_kind(vs) -> str | None:
    if vs.enum_ty is not None:
        return "enum"
    if vs.type_def is not None:
        return "typedef"
    if isinstance(vs.var, N.Function):
        return "func"
    if isinstance(vs.var, N.Var):
        return "var"
    return None


def check_redecl(p, scope, name, kind: str, block_obj: bool = False) -> None:
    """Error if `name` (token) declared as `kind` ("var", "func", "typedef",
    "enum") clashes with what `scope` already holds under that name.
    `block_obj` marks a non-extern block-scope object declaration."""
    vs = scope.vars.get(name.text)
    old = _old_kind(vs) if vs is not None else None
    if old is None:
        return
    if old == kind and kind in ("typedef", "func"):
        return
    if old == "var" and kind == "var":
        if block_obj:
            p.error(f"'{name.text}' redeclared", name)
        return
    p.error(f"'{name.text}' redeclared as different kind of symbol", name)
