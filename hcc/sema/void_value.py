"""Rejects void-typed expressions used where a value is required (C11 6.3.2.2).

    require_value(expr) -> None   error if expr has type void

Callers: binary/logical/unary operators, assignment rhs, conditions and
returns with a value. Void is still legal as an expression statement, as the
operand of a cast to void, and as a comma operand or conditional branch when
both branches are void.
"""
from __future__ import annotations

from hcc import diag
from hcc.typesys import TypeKind

MSG = "void value not ignored as it ought to be"


def is_void(node) -> bool:
    ty = getattr(node, "ty", None)
    return ty is not None and ty.kind == TypeKind.VOID


def require_value(node) -> None:
    if is_void(node):
        diag.error(node, MSG, "sema")
