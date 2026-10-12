"""C11 6.3.2.1p1: an lvalue of incomplete type is only usable as the operand of
unary `&`. Any other consumer of such a value (expression statement, cast,
call argument, operator operand, ...) is a constraint violation.

`check_value_operands(node)` is called by add_type for every freshly typed
expression node; `is_incomplete_object(ty)` is shared with the parser.
"""
from __future__ import annotations

import dataclasses

from hcc import diag
from hcc import nodes as N
from hcc.typesys import TypeKind


def is_incomplete_object(ty) -> bool:
    """True for a struct/union/enum type that has no definition yet."""
    if ty is None:
        return False
    if ty.kind in (TypeKind.STRUCT, TypeKind.UNION):
        t = ty
        while not t.is_complete and t.origin is not None:   # qualified copy made while incomplete
            t = t.origin
        return not t.is_complete
    if ty.kind == TypeKind.ENUM:
        return ty.enum_consts == "fwd"                       # `enum e;` with no body and no fixed type
    return False


def check_value_operands(node) -> None:
    """Reject incomplete-type operands consumed as values by `node` (not `&`)."""
    if isinstance(node, N.AddrOf):
        return
    for f in dataclasses.fields(node):
        v = getattr(node, f.name)
        items = v if isinstance(v, list) else [v]
        for c in items:
            if isinstance(c, N.Expr) and is_incomplete_object(c.ty):
                diag.error(c, f"'{c.ty}' has an incomplete type", "sema")
