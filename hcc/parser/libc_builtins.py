"""Return types of library functions that GCC knows as builtins.

When such a function is called without a visible declaration (for example
`__builtin_memset(...)`, or `memset(...)` with no <string.h>), GCC uses the
builtin signature instead of implicit `int`. Without this, a pointer or
double result is silently truncated. Only the return type is recorded; the
arguments are passed unconverted, as for any prototype-less call.

libc_return_type(name) -> Type | None
"""
from __future__ import annotations

from hcc import typesys as T

_SIZE_T = T.TY_ULONG
_VOID_PTR = T.pointer_to(T.TY_VOID)
_CHAR_PTR = T.pointer_to(T.TY_CHAR)

_TABLE: dict[str, T.Type] = {}


def _add(ty, names: str) -> None:
    for n in names.split():
        _TABLE[n] = ty


_add(_VOID_PTR, "memcpy memmove memset memchr malloc calloc realloc alloca")
_add(_CHAR_PTR, "strcpy strncpy strcat strncat strchr strrchr strstr strpbrk")
_add(_SIZE_T, "strlen strnlen")
_add(T.TY_VOID, "abort exit _Exit free")
_add(T.TY_DOUBLE, "sqrt sin cos tan exp log log10 pow floor ceil fabs atan atan2 copysign")
_add(T.TY_FLOAT, "sqrtf sinf cosf fabsf floorf ceilf powf copysignf")


def libc_return_type(name: str):
    """Return type GCC assumes for builtin library function `name`, or None."""
    return _TABLE.get(name)
