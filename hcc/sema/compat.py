"""Type compatibility (C11 6.2.7), used by _Generic, redeclarations and
__builtin_types_compatible_p. Qualifiers must match at the top level.
"""
from __future__ import annotations

from hcc.typesys import Type, TypeKind


def _root(t: Type) -> Type:
    while t.origin is not None:
        t = t.origin
    return t


def is_compatible(a: Type, b: Type) -> bool:
    if a is b:
        return True
    if a.is_const != b.is_const or a.is_volatile != b.is_volatile:
        return False
    if a.kind != b.kind:
        return False
    k = a.kind
    if k in (TypeKind.CHAR, TypeKind.SHORT, TypeKind.INT, TypeKind.LONG, TypeKind.LLONG,
             TypeKind.INT128, TypeKind.BOOL):
        return a.is_unsigned == b.is_unsigned
    if k in (TypeKind.FLOAT16, TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LDOUBLE, TypeKind.FLOAT128,
             TypeKind.VOID):
        return True
    if k == TypeKind.PTR:
        return is_compatible(a.base, b.base)
    if k == TypeKind.FUNC:
        if not is_compatible(a.return_ty, b.return_ty):
            return False
        if not a.has_proto or not b.has_proto:
            return True
        if a.is_variadic != b.is_variadic or len(a.params) != len(b.params):
            return False
        return all(is_compatible(x.unqual(), y.unqual()) for x, y in zip(a.params, b.params))
    if k == TypeKind.ARRAY:
        if not is_compatible(a.base, b.base):
            return False
        return a.array_len < 0 or b.array_len < 0 or a.array_len == b.array_len
    if k in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ENUM):
        return _root(a) is _root(b)
    return False
