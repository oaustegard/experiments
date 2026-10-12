"""Full prototypes of library builtins that GCC knows even without a header.

A file-scope prototype of such a name must be compatible with the builtin's
signature (`int strcmp(void);` is an error), while an unprototyped
declaration is always fine. libc_prototype(name) -> Type | None.
"""
from __future__ import annotations

from hcc import typesys as T


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
