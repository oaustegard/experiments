"""GCC: declaring a name whose type is unavailable is an error.

A struct/union marked __attribute__((unavailable)) may not be named by any
declarator (object, typedef, function return type), even through pointers and
arrays. Only the declaration is checked here; uses of the names are checked by
core.check_unavailable_type() from primary.py.
"""
from __future__ import annotations

from hcc import typesys as T


class UnavailableDeclMixin:
    def check_unavailable_declarator(self, ty, tok) -> None:
        """Error if the type a declarator names reaches an unavailable struct/union."""
        t = ty
        while t is not None:
            self.check_unavailable_type(t, tok)
            if t.kind in (T.TypeKind.PTR, T.TypeKind.ARRAY):
                t = t.base
            elif t.kind == T.TypeKind.FUNC:
                t = t.return_ty
            else:
                break
