"""Incomplete object types used where a value (or a size) is needed.

An lvalue of incomplete struct/union/enum type may not be converted to an
rvalue (C11 6.3.2.1p2): `s;`, `f(s)`, `int a[s]`, ... are constraint errors.
Taking its address or naming it in `&`/`sizeof` is still fine, which is why
this check is applied only at the value-use sites, not in add_type().
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T


def is_incomplete_object(ty) -> bool:
    """True for struct/union types still missing their body and for enum
    forward references (`enum e` without a body or fixed underlying type,
    which can never be completed)."""
    if ty is None:
        return False
    if ty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION):
        return not ty.is_complete
    if ty.kind == T.TypeKind.ENUM:
        return ty.enum_consts == "fwd"
    return False


def check_value_use(parser, node) -> None:
    """Report an error when `node` (already typed) is an incomplete object used as a value."""
    ty = getattr(node, "ty", None)
    if not is_incomplete_object(ty):
        return
    if isinstance(node, N.VarRef):
        parser.error(f"'{node.var.name}' has an incomplete type", node)
    else:
        parser.error("invalid use of undefined type", node)
