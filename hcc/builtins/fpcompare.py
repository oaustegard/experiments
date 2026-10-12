"""Quiet floating comparisons: __builtin_is{unordered,greater,greaterequal,less,
lessequal,lessgreater}(a, b). Codegen (floats.py fp_cmp_builtin) sets the result
from the ucomi flags."""
from __future__ import annotations

from hcc import diag
from hcc import typesys as T
from hcc.builtins import builtin_codegen, builtin_parser
from hcc.sema.conversions import new_cast

_OPS = ("isunordered", "isgreater", "isgreaterequal", "isless", "islessequal", "islessgreater")


@builtin_parser(*(f"__builtin_{op}" for op in _OPS))
def _parse_fpcmp(self, tok):
    """Both operands are widened to long double if either is, else to double
    (exact for float). An integer operand is converted to the other operand's
    floating type, as in GCC; only if neither operand is floating is it an error."""
    name = tok.text[len("__builtin_"):]
    args = self._args(2)
    if not any(T.is_flonum(a.ty) for a in args):
        diag.error(tok, f"non-floating-point arguments in call to function '{tok.text}'")
    wide = any(a.ty.kind == T.TypeKind.LDOUBLE for a in args)
    ty = T.TY_LDOUBLE if wide else T.TY_DOUBLE
    return self._call(tok, name, [new_cast(a, ty) for a in args], T.TY_INT)


@builtin_codegen(*_OPS)
def _gen_fpcmp(self, node) -> None:
    self.fp_cmp_builtin(node)
