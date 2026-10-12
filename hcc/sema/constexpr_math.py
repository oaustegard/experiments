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
}
_BINARY = {"atan2": math.atan2, "pow": math.pow, "fmod": math.fmod}
_NAN = ("nan", "nanf", "nanl")


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
    name = callee.var.name
    args = node.args
    if name in _NAN and len(args) == 1 and _is_string(args[0]):
        return math.nan
    try:
        if name in _UNARY and len(args) == 1:
            return float(_UNARY[name](eval_float(args[0])))
        if name in _BINARY and len(args) == 2:
            return float(_BINARY[name](eval_float(args[0]), eval_float(args[1])))
    except (ValueError, OverflowError, ZeroDivisionError):
        return None         # domain/range error: leave it to the runtime call
    return None
