"""Sema rules for _Complex operands (C99 6.5, Annex G).

Operands of complex arithmetic are converted to the common complex type by
usual_arith_conv (real operands become `_Complex T` with a zero imaginary part).
Only + - * / (and == !=) are defined on complex values here.
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N

COMPLEX_OPS = ("+", "-", "*", "/", "==", "!=")


def complex_check_op(n: N.Binary) -> None:
    if n.op not in COMPLEX_OPS:
        diag.error(n, f"invalid operands to binary '{n.op}' (complex operand)", "sema")


def complex_static_parts(expr) -> tuple[float, float]:
    """(real, imag) of a constant complex-valued expression, for static data.
    Raises constexpr.NotConstant for anything else."""
    from hcc.sema.constexpr import NotConstant, eval_float
    from hcc.typesys import is_complex
    if isinstance(expr, N.ComplexLit):
        return expr.re, expr.im
    if isinstance(expr, N.Cast):
        if is_complex(expr.operand.ty):
            return complex_static_parts(expr.operand)
        return eval_float(expr.operand), 0.0          # real -> complex: imag = +0.0
    if isinstance(expr, N.Binary) and expr.op in ("+", "-") and is_complex(expr.ty):
        (ar, ai), (br, bi) = complex_static_parts(expr.lhs), complex_static_parts(expr.rhs)
        if expr.op == "+":
            return ar + br, ai + bi
        return ar - br, ai - bi
    if isinstance(expr, N.Unary) and expr.op == "-" and is_complex(expr.ty):
        re, im = complex_static_parts(expr.operand)
        return -re, -im
    if expr.ty is not None and not is_complex(expr.ty):   # real initializer, not yet converted
        return eval_float(expr), 0.0
    raise NotConstant


def complex_compare_const(node) -> int:
    """Value of `a == b` / `a != b` on constant complex operands (0 or 1)."""
    (ar, ai), (br, bi) = complex_static_parts(node.lhs), complex_static_parts(node.rhs)
    eq = ar == br and ai == bi               # NaN parts compare unequal, as in C
    return int(eq) if node.op == "==" else int(not eq)
