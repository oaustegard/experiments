"""Type compatibility (C11 6.2.7), used by _Generic, redeclarations and
__builtin_types_compatible_p. Qualifiers must match at the top level.
"""
from __future__ import annotations

from hcc.typesys import Type, TypeKind


_ARRAYISH = (TypeKind.ARRAY, TypeKind.VLA)


_PROMOTED = (TypeKind.CHAR, TypeKind.SHORT, TypeKind.BOOL, TypeKind.FLOAT16, TypeKind.FLOAT)


def _promotion_stable(t: Type) -> bool:
    """True when the default argument promotions leave parameter type `t` unchanged."""
    return t.kind not in _PROMOTED


def _root(t: Type) -> Type:
    while t.origin is not None:
        t = t.origin
    return t


def _ret_unqualified(t: Type) -> Type:
    """Return type with top-level const/volatile dropped (no copy when none present)."""
    if t is None or not (t.is_const or t.is_volatile):
        return t
    t = t.copy()
    t.is_const = t.is_volatile = False
    return t


def _enum_as_int(t: Type) -> Type:
    """An enum type is compatible with its underlying integer type (C11 6.7.2.2):
    the signed/unsigned integer type of the same size (char, short, int or long)."""
    kind = {1: TypeKind.CHAR, 2: TypeKind.SHORT, 8: TypeKind.LONG}.get(t.size, TypeKind.INT)
    return Type(kind, t.size, t.align, bool(t.is_unsigned))


def is_compatible(a: Type, b: Type) -> bool:
    if a is b:
        return True
    if a.is_const != b.is_const or a.is_volatile != b.is_volatile or a.is_atomic != b.is_atomic:
        return False
    if a.kind in _ARRAYISH and b.kind in _ARRAYISH:
        # A VLA (or `[*]`) is compatible with any array of a compatible element type.
        if not is_compatible(a.base, b.base):
            return False
        if a.kind == TypeKind.VLA or b.kind == TypeKind.VLA:
            return True
        return a.array_len < 0 or b.array_len < 0 or a.array_len == b.array_len
    if a.kind != b.kind:
        if a.kind == TypeKind.ENUM and b.kind != TypeKind.ENUM:
            return is_compatible(_enum_as_int(a), b)
        if b.kind == TypeKind.ENUM and a.kind != TypeKind.ENUM:
            return is_compatible(a, _enum_as_int(b))
        return False
    if a.is_complex or b.is_complex:      # _Complex T: compatible iff the element types are
        return a.is_complex and b.is_complex and is_compatible(a.base, b.base)
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
        # GCC ignores top-level const/volatile on a function's return type
        # (`int f(); const int f() {...}` is a valid redeclaration); _Atomic is kept.
        if not is_compatible(_ret_unqualified(a.return_ty), _ret_unqualified(b.return_ty)):
            return False
        if not a.has_proto or not b.has_proto:
            # Unprototyped vs prototyped (C11 6.7.6.3p15): the prototype must not be
            # variadic and each parameter must be unchanged by default argument promotions.
            proto = b if not a.has_proto else a
            if not proto.has_proto:
                return True
            return not proto.is_variadic and all(_promotion_stable(p) for p in proto.params)
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
