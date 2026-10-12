"""Argument compatibility for struct/union parameters (C11 6.5.2.2p7).

A struct or union parameter may only receive a compatible struct or union
value, and a scalar parameter may not receive a struct or union value.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.compat import is_compatible
from hcc.typesys import is_struct_like


def check_struct_arg(parser, fn, index: int, pty, aty, loc) -> None:
    if pty is None or aty is None:
        return
    if pty.is_complex or aty.is_complex or T.is_dyn_struct(pty) or T.is_dyn_struct(aty):
        return
    if not (is_struct_like(pty) or is_struct_like(aty)):
        return
    if is_struct_like(pty) and is_struct_like(aty) and is_compatible(pty.unqual(), aty.unqual()):
        return
    name = fn.var.name if isinstance(fn, N.VarRef) and isinstance(fn.var, N.Function) else "function"
    parser.error("incompatible type for argument %d of '%s'" % (index + 1, name), loc)
