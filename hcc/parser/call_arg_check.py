"""Argument compatibility for struct/union parameters (C11 6.5.2.2p7).

A struct or union parameter may only receive a compatible struct or union
value, and a scalar parameter may not receive a struct or union value.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.compat import is_compatible
from hcc.typesys import is_dyn_struct, is_struct_like


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


def check_pointer_float_arg(parser, fn, index: int, pty, aty, loc) -> None:
    """A pointer never converts to or from a floating type (C11 6.5.16.1p1).

    Arrays and functions decay to pointers first; complex and vector operands are
    handled by their own conversions and are not checked here.
    """
    if pty is None or aty is None or pty.is_complex or aty.is_complex:
        return
    if pty.is_vector or aty.is_vector or is_dyn_struct(pty) or is_dyn_struct(aty):
        return
    p_ptr = pty.kind == T.TypeKind.PTR
    a_ptr = aty.kind in (T.TypeKind.PTR, T.TypeKind.ARRAY, T.TypeKind.FUNC)
    if not ((p_ptr and T.is_flonum(aty)) or (T.is_flonum(pty) and a_ptr)):
        return
    name = fn.var.name if isinstance(fn, N.VarRef) and isinstance(fn.var, N.Function) else "function"
    parser.error("incompatible type for argument %d of '%s'" % (index + 1, name), loc)
