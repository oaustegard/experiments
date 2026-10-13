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
_add(T.TY_DOUBLE, "asin acos sinh cosh tanh log2 exp2 cbrt trunc round fmod hypot fmin fmax")
_add(T.TY_FLOAT, "sqrtf sinf cosf fabsf floorf ceilf powf copysignf")
_add(T.TY_LDOUBLE, "sqrtl sinl cosl tanl expl logl log10l powl floorl ceill fabsl atanl atan2l copysignl")
# C99 math functions of one or two floating arguments (double, f and l variants).
for _base in ("acosh asinh atanh erf erfc expm1 log1p tgamma lgamma nearbyint rint "
              "remainder fdim acos asin cbrt cosh exp2 fmod hypot log2 sinh tanh "
              "frexp ldexp scalbn scalbln logb modf remquo nan").split():
    _add(T.TY_DOUBLE, _base)
    _add(T.TY_FLOAT, _base + "f")
    _add(T.TY_LDOUBLE, _base + "l")
_add(T.TY_INT, "ilogb ilogbf ilogbl")
_add(T.TY_LONG, "lrint lround lrintf lroundf lrintl lroundl")
_add(T.TY_LLONG, "llrint llround llrintf llroundf llrintl llroundl")
_add(T.TY_INT, "abs memcmp strcmp strncmp")
_add(T.TY_LONG, "labs imaxabs")
_add(T.TY_LLONG, "llabs")

_add(T.TY_FLOAT, "tanf expf logf log10f log2f exp2f atanf atan2f asinf acosf fmodf hypotf")
_add(T.TY_DOUBLE, "asinh acosh atanh erf erfc expm1 lgamma log1p logb tgamma")
_add(T.TY_FLOAT, "asinhf acoshf atanhf erff erfcf expm1f lgammaf log1pf logbf tgammaf")
_add(T.TY_LDOUBLE, "asinhl acoshl atanhl erfl erfcl expm1l lgammal log1pl logbl tgammal")
_add(T.TY_DOUBLE, "fdim remainder ldexp scalbn scalbln frexp modf remquo")
_add(T.TY_FLOAT, "fdimf remainderf ldexpf scalbnf scalblnf frexpf modff remquof")
_add(T.TY_LDOUBLE, "fdiml remainderl ldexpl scalbnl scalblnl frexpl modfl remquol")
_add(T.TY_INT, "ilogb ilogbf ilogbl")
_add(T.TY_LONG, "lrint lround lrintf lroundf lrintl lroundl")
_add(T.TY_LLONG, "llrint llround llrintf llroundf llrintl llroundl")
_add(T.TY_DOUBLE, "fma nextafter")
_add(T.TY_FLOAT, "fmaf nextafterf")
_add(T.TY_LDOUBLE, "fmal nextafterl")

# Builtin math functions take fixed floating parameters: GCC converts the
# arguments to them, so `__builtin_log2(an_unsigned_long)` works. Only the
# names listed here get a prototype; the rest keep the unprototyped call.
_MATH_ARGS: dict[str, list[T.Type]] = {}
for _names, _n in (("sqrt sin cos tan exp log log10 floor ceil fabs atan asin acos sinh cosh tanh "
                    "log2 exp2 cbrt trunc round acosh asinh atanh erf erfc expm1 log1p tgamma lgamma "
                    "nearbyint rint", 1),
                   ("pow atan2 copysign fmod hypot fmin fmax remainder fdim", 2)):
    for _nm in _names.split():
        _MATH_ARGS[_nm] = [T.TY_DOUBLE] * _n
        _MATH_ARGS[_nm + "f"] = [T.TY_FLOAT] * _n


def libc_return_type(name: str):
    """Return type GCC assumes for builtin library function `name`, or None."""
    return _TABLE.get(name)


_FLOAT_RET = (T.TypeKind.FLOAT, T.TypeKind.DOUBLE, T.TypeKind.LDOUBLE)


def float128_libm_call(spelled: str):
    """`__builtin_<fn>f128[x](...)` returns _Float128, not long double. Given the
    spelling `<fn>f128` / `<fn>f128x` (after `__builtin_`), return
    (`<fn>f128`, _Float128, [_Float128]*arity) so the call uses the binary128 libm
    function, or None when the spelling is not a floating f128 builtin."""
    for sfx in ("f128x", "f128"):
        if spelled.endswith(sfx) and len(spelled) > len(sfx):
            base = spelled[:-len(sfx)]
            brty = _TABLE.get(base)
            if brty is None or brty.kind not in _FLOAT_RET:
                return None
            arity = _libm_arity(base)
            params = None if arity is None else [T.TY_FLOAT128] * arity
            return base + "f128", T.TY_FLOAT128, params
    return None


# Floating libm builtins taking two operands (base name, before any f/l suffix);
# every other entry takes one.
_LIBM_TWO_ARG = frozenset(
    "pow copysign atan2 fmod hypot fmin fmax nextafter remainder fdim".split())
_LIBM_THREE_ARG = frozenset("fma fmaf fmal".split())
# Floating-return builtins whose other operands are not floating (int exponent,
# pointer out-parameter, char* tag): left unprototyped rather than given a guessed arity.
_LIBM_UNPROTOTYPED = frozenset(
    "frexp ldexp scalbn scalbln modf remquo nan ilogb logb lrint lround llrint llround".split())
_LIBM_FLOAT_KINDS = (T.TypeKind.FLOAT, T.TypeKind.DOUBLE, T.TypeKind.LDOUBLE)


def _libm_arity(name: str):
    """Operand count of libm builtin `name` (None: not a prototyped floating builtin)."""
    base = name[:-1] if name[-1:] in ("f", "l") else name
    if name in _LIBM_THREE_ARG:
        return 3
    if name in _LIBM_TWO_ARG or base in _LIBM_TWO_ARG:
        return 2
    if name in _LIBM_UNPROTOTYPED or base in _LIBM_UNPROTOTYPED:
        return None
    return 1


def libm_param_types(name: str, rty):
    """Parameter types GCC gives an implicit call of libm builtin `name` (a
    floating function with return type `rty`): each operand has the function's
    own type, so float arguments are not promoted to double. None when the
    name has no such builtin prototype."""
    if rty is None or rty.kind not in _LIBM_FLOAT_KINDS or _TABLE.get(name) is not rty:
        return None
    arity = _libm_arity(name)
    return None if arity is None else [rty] * arity


# __builtin_powi{f,,l}(x, n): integer power, expanded by libgcc as __powi{sf,df,xf}2.
# Maps the builtin's libm-style name to (libgcc helper, C type of x and result).
POWI_HELPERS = {
    "powif": ("__powisf2", T.TY_FLOAT),
    "powi": ("__powidf2", T.TY_DOUBLE),
    "powil": ("__powixf2", T.TY_LDOUBLE),
}


def libc_param_types(name: str):
    """Parameter types of builtin math function `name` (a new list), or None."""
    ps = _MATH_ARGS.get(name)
    return None if ps is None else list(ps)


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
