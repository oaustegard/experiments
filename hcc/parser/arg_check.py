"""Argument/parameter compatibility checks for calls to prototyped functions.

check_call_arg(parser, arg, pty) rejects the pairings C forbids outright
(gcc: "incompatible type for argument N"):
  * struct/union <-> non-aggregate scalar, or two incompatible struct types
  * floating/complex <-> pointer (either direction)
Integer <-> pointer and pointer-sign mismatches are left to conversion
(gcc only warns about those under -fpermissive).
"""
from __future__ import annotations

from hcc.sema.compat import is_compatible
from hcc.typesys import TypeKind, is_complex, is_flonum, is_struct_like

_PTR_LIKE = (TypeKind.PTR, TypeKind.ARRAY, TypeKind.FUNC, TypeKind.VLA)


def _is_float_like(t) -> bool:
    return is_flonum(t) or is_complex(t)


def check_call_arg(parser, arg, pty) -> None:
    src = arg.ty
    if src is None or pty is None:
        return
    if src.is_vector or pty.is_vector or src.is_complex or pty.is_complex:
        return
    if is_struct_like(pty) or is_struct_like(src):
        if is_struct_like(pty) and is_struct_like(src):
            if not is_compatible(src, pty):
                parser.error("incompatible type for argument", arg.loc)
        else:
            parser.error("incompatible type for argument", arg.loc)
        return
    if pty.kind in _PTR_LIKE and _is_float_like(src):
        parser.error("incompatible type for argument", arg.loc)
    elif _is_float_like(pty) and src.kind in _PTR_LIKE:
        parser.error("incompatible type for argument", arg.loc)
