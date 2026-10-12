"""Return types of library functions that GCC knows as builtins.

When such a function is called without a visible declaration (for example
`__builtin_memset(...)`, or `memset(...)` with no <string.h>), GCC uses the
builtin signature instead of implicit `int`. Without this, a pointer or
double result is silently truncated. Only the return type is recorded; the
arguments are passed unconverted, as for any prototype-less call.

libc_return_type(name) -> Type | None
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.conversions import new_cast

_SIZE_T = T.TY_ULONG
_VOID_PTR = T.pointer_to(T.TY_VOID)
_CHAR_PTR = T.pointer_to(T.TY_CHAR)

_TABLE: dict[str, T.Type] = {}


def _add(ty, names: str) -> None:
    for n in names.split():
        _TABLE[n] = ty


_add(_VOID_PTR, "memcpy memmove mempcpy memset memchr malloc calloc realloc alloca")
_add(_CHAR_PTR, "strcpy stpcpy strncpy strcat strncat strchr strrchr strstr strpbrk strdup strndup")
_add(_SIZE_T, "strlen strnlen")
_add(T.TY_VOID, "abort exit _Exit free")
_add(T.TY_DOUBLE, "sqrt sin cos tan exp log log10 pow floor ceil fabs atan atan2 copysign")
_add(T.TY_FLOAT, "sqrtf sinf cosf fabsf floorf ceilf powf copysignf")
_add(T.TY_LDOUBLE, "sqrtl sinl cosl tanl expl logl log10l powl floorl ceill fabsl atanl atan2l copysignl")
_add(T.TY_INT, "abs")
_add(T.TY_LONG, "labs imaxabs")
_add(T.TY_LLONG, "llabs")


def libc_return_type(name: str):
    """Return type GCC assumes for builtin library function `name`, or None."""
    return _TABLE.get(name)


# Floating libm builtins taking two operands; every other entry takes one.
_LIBM_TWO_ARG = frozenset(
    "pow copysign atan2 powf copysignf atan2f powl copysignl atan2l".split())
_LIBM_FLOAT_KINDS = (T.TypeKind.FLOAT, T.TypeKind.DOUBLE, T.TypeKind.LDOUBLE)


def libm_param_types(name: str, rty):
    """Parameter types GCC gives an implicit call of libm builtin `name` (a
    floating function with return type `rty`): each operand has the function's
    own type, so float arguments are not promoted to double. None when the
    name has no such builtin prototype."""
    if rty is None or rty.kind not in _LIBM_FLOAT_KINDS or _TABLE.get(name) is not rty:
        return None
    return [rty] * (2 if name in _LIBM_TWO_ARG else 1)


# __builtin_powi{f,,l}(x, n): integer power, expanded by libgcc as __powi{sf,df,xf}2.
# Maps the builtin's libm-style name to (libgcc helper, C type of x and result).
POWI_HELPERS = {
    "powif": ("__powisf2", T.TY_FLOAT),
    "powi": ("__powidf2", T.TY_DOUBLE),
    "powil": ("__powixf2", T.TY_LDOUBLE),
}


# Integer abs() functions that GCC expands inline (bit width of the operand).
_INT_ABS = {"abs": 32, "labs": 64, "llabs": 64, "imaxabs": 64}


def int_abs_call(loc, name: str, arg):
    """If the call `name(arg)` is a builtin integer abs, return the
    BuiltinCall node (GCC expands it inline even when the program declares
    or defines the function itself); otherwise None."""
    bits = _INT_ABS.get(name)
    if bits is None:
        return None
    ty = T.TY_INT if bits == 32 else T.TY_LONG
    return N.BuiltinCall(loc, f"abs{bits}", [new_cast(arg, ty)], None, ty=ty)


def fabs_float_call(loc, name: str, arg):
    """`fabs(x)` with a float x is expanded by GCC as an inline sign-bit clear on
    the float (no widening to double), so a signaling NaN stays signaling. Returns
    the float-typed BuiltinCall, or None when the call is not of that form."""
    if name != "fabs" or arg.ty is None or arg.ty.kind != T.TypeKind.FLOAT:
        return None
    return N.BuiltinCall(loc, "fabsf", [arg], None, ty=T.TY_FLOAT)
