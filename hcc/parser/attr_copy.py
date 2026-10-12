"""`copy(expr)` member attributes and member queries for __builtin_has_attribute.

GCC's `copy(expr)` gives a declaration the attributes of the expression's
decl. hcc only tracks `packed` and `aligned` on members, so:
  * a member access `x.m` / `p->m` is packed when the member itself is packed
    or the struct it is accessed through is packed;
  * any other expression is packed when its type, after stripping pointers and
    arrays, is a packed struct or union.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T


def struct_packed(ty) -> bool:
    """True when `ty` (through pointers/arrays) is a packed struct or union."""
    while ty is not None and ty.kind in (T.TypeKind.PTR, T.TypeKind.ARRAY):
        ty = ty.base
    return ty is not None and ty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION) and bool(ty.is_packed)


def expr_is_packed(e) -> bool:
    if isinstance(e, N.Member_):
        if e.member.is_packed or struct_packed(e.operand.ty):
            return True
    return struct_packed(getattr(e, "ty", None))


def member_has_attr(e, want: str) -> bool:
    """Attribute `want` on a member-access expression `e` (Member_)."""
    m = e.member
    if want == "packed":
        return bool(m.is_packed or struct_packed(e.operand.ty))
    if want == "aligned":
        return m.explicit_align > 0
    return False


def apply_copy_expr(parser, args_start: int, attr) -> None:
    """`copy(expr)` on a declaration: record `packed` when expr's attributes include it.

    `args_start` is the index of the '(' that opens the copy() arguments."""
    save = parser.pos
    parser.pos = args_start + 1
    try:
        e = parser.typed(parser.assign())
    finally:
        parser.pos = save
    if expr_is_packed(e):
        attr.is_packed = True
