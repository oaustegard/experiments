"""Compile-time folding of pure libm calls (GNU extension used in static initializers).

    fold_math_call(node, eval_float) -> float | None
        Value of `atan(1.0)`, `nan("")`, ... when the callee is a known pure math
        function and every argument is constant; None otherwise (caller raises
        NotConstant). `eval_float` evaluates a constant floating argument.
"""
from __future__ import annotations

import math

from hcc import nodes as N

_UNARY = {
    "atan": math.atan, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "sinh": math.sinh, "cosh": math.cosh,
    "tanh": math.tanh, "exp": math.exp, "log": math.log, "log10": math.log10,
    "sqrt": math.sqrt, "fabs": math.fabs, "floor": math.floor, "ceil": math.ceil,
    "acosh": math.acosh, "asinh": math.asinh, "atanh": math.atanh, "cbrt": lambda x: _cbrt(x),
    "erf": math.erf, "erfc": math.erfc, "exp2": lambda x: 2.0 ** x,
    "expm1": math.expm1, "log1p": math.log1p, "log2": math.log2,
    "logb": lambda x: math.floor(math.log2(abs(x))) if x else -math.inf,
    "tgamma": math.gamma, "lgamma": lambda x: math.lgamma(x),
    "trunc": math.trunc, "rint": round,
}
def _cbrt(x: float) -> float:
    """Cube root that is exact for exact cubes (libm does; math.cbrt may be 1 ulp off)."""
    r = math.cbrt(x)
    n = round(r)
    return float(n) if n ** 3 == x else r


_BINARY = {
    "atan2": math.atan2, "pow": math.pow, "fmod": math.fmod,
    "copysign": math.copysign, "hypot": math.hypot, "remainder": math.remainder,
    "nextafter": math.nextafter, "fdim": lambda a, b: a - b if a > b else 0.0,
    "ldexp": lambda a, n: math.ldexp(a, int(n)), "scalbn": lambda a, n: math.ldexp(a, int(n)),
    "scalbln": lambda a, n: math.ldexp(a, int(n)),
}
_NAN = ("nan", "nanf", "nanl")


_FLOAT_SUFFIXES = ("f32x", "f64x", "f128x", "f32", "f64", "f128", "f", "l")


def _base_name(name: str) -> str:
    """`acoshl` / `expm1f` / `acoshf64x` -> `acosh` / `expm1` / `acosh`: the C99 and
    GNU _FloatN variants of a libm function fold like the double version."""
    if name in _KNOWN:
        return name
    for sfx in _FLOAT_SUFFIXES:
        if name.endswith(sfx) and name[:-len(sfx)] in _KNOWN:
            return name[:-len(sfx)]
    return name


def _strip_casts(e):
    while isinstance(e, N.Cast):
        e = e.operand
    return e


def _is_string(e) -> bool:
    e = _strip_casts(e)
    if isinstance(e, N.StrLit):
        return True
    return isinstance(e, N.VarRef) and getattr(e.var, "is_string_literal", False)


def fold_math_call(node, eval_float):
    if not isinstance(node, N.Call):
        return None
    callee = _strip_casts(node.func)
    if not isinstance(callee, N.VarRef):
        return None
    name = _base_name(callee.var.name)
    args = node.args
    if callee.var.name in _NAN and len(args) == 1 and _is_string(args[0]):
        return math.nan
    try:
        if name in _UNARY and len(args) == 1:
            return float(_UNARY[name](eval_float(args[0])))
        if name in _BINARY and len(args) == 2:
            return float(_BINARY[name](eval_float(args[0]), eval_float(args[1])))
    except (ValueError, OverflowError, ZeroDivisionError):
        return None         # domain/range error: leave it to the runtime call
    return None


def _raise():
    raise ValueError("ilogb(0) is FP_ILOGB0")


def _round_away(x: float) -> int:
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


# libm functions whose result has an integer type (lround, ilogb, ...).
_INT_RESULT = {
    "ilogb": lambda x: math.frexp(x)[1] - 1 if x else _raise(),
    "lround": _round_away, "llround": _round_away,
    "lrint": round, "llrint": round,
}


_KNOWN = _UNARY.keys() | _BINARY.keys() | _INT_RESULT.keys()


def fold_int_call(node, eval_float):
    """Integer value of an integer-typed libm call such as `lround(42.25)`, or None."""
    if not isinstance(node, N.Call):
        return None
    callee = _strip_casts(node.func)
    if not isinstance(callee, N.VarRef):
        return None
    name = _base_name(callee.var.name)
    if name not in _INT_RESULT or len(node.args) != 1:
        return None
    try:
        return int(_INT_RESULT[name](eval_float(node.args[0])))
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
