"""Const-qualified parts of aggregates (C11 6.3.2.1p1).

A struct or union object is not a modifiable lvalue when it has, directly or
through nested members and array elements, a const-qualified member. Its own
type can carry no const qualifier (`volatile struct S t;`), so the check walks
the member list.
"""
from __future__ import annotations

from hcc.typesys import Type, TypeKind


def _complete(ty: Type) -> Type:
    """A qualified copy made while the tag was incomplete keeps an empty member
    list: follow `origin` to the type that holds the real members."""
    while not ty.is_complete and ty.origin is not None:
        ty = ty.origin
    return ty


def has_const_member(ty: Type | None) -> bool:
    """True if a struct/union `ty` has a const-qualified member, recursively."""
    if ty is None or ty.kind not in (TypeKind.STRUCT, TypeKind.UNION):
        return False
    # a struct cannot contain itself by value, so this recursion terminates
    for m in _complete(ty).members:
        if _const_part(m.ty):
            return True
    return False


def _const_part(ty: Type | None) -> bool:
    if ty is None:
        return False
    if ty.is_const:
        return True
    if ty.kind == TypeKind.ARRAY:
        return _const_part(ty.base)
    if ty.kind in (TypeKind.STRUCT, TypeKind.UNION):
        return has_const_member(ty)
    return False
