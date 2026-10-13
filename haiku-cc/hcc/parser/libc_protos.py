"""Full prototypes of library builtins that GCC knows even without a header.

A file-scope prototype of such a name must be compatible with the builtin's
signature (`int strcmp(void);` is an error), while an unprototyped
declaration is always fine. libc_prototype(name) -> Type | None.
"""
from __future__ import annotations

from hcc import typesys as T
from hcc.typesys import FLOAT_KINDS, INTEGER_KINDS, Type, TypeKind


def _const_ptr(base: T.Type) -> T.Type:
    c = base.copy()
    c.is_const = True
    return T.pointer_to(c)


def _proto(ret: T.Type, params: list) -> T.Type:
    fty = T.func_type(ret)
    fty.params = params
    fty.has_proto = True
    return fty


_SIZE_T = T.TY_ULONG
_CONST_CHAR_PTR = _const_ptr(T.TY_CHAR)
_CONST_VOID_PTR = _const_ptr(T.TY_VOID)

_PROTOS = {
    "strcmp": _proto(T.TY_INT, [_CONST_CHAR_PTR, _CONST_CHAR_PTR]),
    "strncmp": _proto(T.TY_INT, [_CONST_CHAR_PTR, _CONST_CHAR_PTR, _SIZE_T]),
    "memcmp": _proto(T.TY_INT, [_CONST_VOID_PTR, _CONST_VOID_PTR, _SIZE_T]),
}


def libc_prototype(name: str):
    """The builtin prototype of `name` (shared, do not mutate), or None."""
    return _PROTOS.get(name)


def _shape_class(t: Type) -> object:
    """Coarse class of a parameter/return type: all integers agree, all floats
    agree, all pointers agree (pointee and qualifiers ignored)."""
    if t.kind in INTEGER_KINDS:
        return "int"
    if t.kind in FLOAT_KINDS:
        return "float"
    if t.kind in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.VLA):
        return "ptr"
    return t.kind


def builtin_shape_matches(builtin: Type, ty: Type) -> bool:
    """GCC only warns (-Wbuiltin-declaration-mismatch) when a file-scope prototype
    differs from a builtin in pointee qualifiers or integer signedness; it is an
    error only when the arity, variadicity or the kind of a return/parameter
    differs. True when `ty` is not such a mismatch against `builtin`."""
    if builtin.is_variadic != ty.is_variadic or len(builtin.params) != len(ty.params):
        return False
    if _shape_class(builtin.return_ty) != _shape_class(ty.return_ty):
        return False
    return all(_shape_class(x) == _shape_class(y) for x, y in zip(builtin.params, ty.params))
