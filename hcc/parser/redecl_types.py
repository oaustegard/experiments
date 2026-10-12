"""Type check for an object redeclared in the same scope (C11 6.7p3, 6.2.7).

A second declaration of an ordinary object name in one scope must have a
compatible type; otherwise gcc reports "conflicting types".  Types that involve
a struct/union/enum tag are not compared here (tag identity across forward
declarations and later definitions is not tracked by is_compatible).
"""
from __future__ import annotations

from hcc.sema.compat import is_compatible
from hcc.typesys import Type, TypeKind

_TAGGED = (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ENUM)


def _has_tag(t: Type, depth: int = 0) -> bool:
    if depth > 16 or t is None:
        return False
    if t.kind in _TAGGED:
        return t.tag is not None     # an anonymous struct/union/enum has no tag to forward-declare
    if t.kind in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.VLA):
        return _has_tag(t.base, depth + 1)
    if t.kind == TypeKind.FUNC:
        if _has_tag(t.return_ty, depth + 1):
            return True
        return any(_has_tag(p, depth + 1) for p in t.params)
    return False


def redecl_conflicts(prev: Type, new: Type) -> bool:
    """True when two declarations of one object name cannot describe the same object."""
    if prev is new or _has_tag(prev) or _has_tag(new):
        return False
    return not is_compatible(prev, new)


class RedeclTypesMixin:

    def check_redecl_type(self, prev_var, name: str, ty: Type, loc) -> None:
        """Error if a redeclaration of `name` in the current scope conflicts with prev_var."""
        if prev_var is None or not hasattr(prev_var, "ty"):
            return
        if redecl_conflicts(prev_var.ty, ty):
            self.error(f"conflicting types for '{name}'", loc)


def composite_redecl(prev: Type, new: Type, depth: int = 0) -> Type:
    """Composite type of two compatible declarations of one object (C11 6.2.7p3).

    An incomplete array takes the length of a complete one at any level of
    pointer/array nesting: `int (*[])[]` and `int (*[])[5]` compose to
    `int (*[])[5]`. Anything else keeps `prev`."""
    if depth > 16 or prev is new or prev is None or new is None:
        return prev
    if prev.kind == TypeKind.ARRAY and new.kind == TypeKind.ARRAY and prev.base is not None \
            and new.base is not None:
        if prev.array_len < 0 and new.array_len >= 0:
            out = new.copy()
        elif new.array_len < 0 or prev.array_len >= 0:
            out = prev.copy()
        else:
            return prev
        out.base = composite_redecl(prev.base, new.base, depth + 1)
        return out
    if prev.kind == TypeKind.PTR and new.kind == TypeKind.PTR and prev.base is not None \
            and new.base is not None:
        base = composite_redecl(prev.base, new.base, depth + 1)
        if base is prev.base:
            return prev
        out = prev.copy()
        out.base = base
        return out
    return prev
