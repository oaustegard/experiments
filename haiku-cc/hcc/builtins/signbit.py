"""__builtin_signbit{,f,l}(x)."""
from __future__ import annotations

from hcc import typesys as T
from hcc.builtins import builtin_codegen, builtin_parser
from hcc.sema.conversions import new_cast


@builtin_parser("__builtin_signbit", "__builtin_signbitf", "__builtin_signbitl")
def _parse_signbit(self, tok):
    """__builtin_signbit(x): operand widened exactly to double (keeps the
    sign bit, also for NaN and float/long double); codegen reads bit 63."""
    args = [new_cast(a, T.TY_DOUBLE) for a in self._args(1)]
    return self._call(tok, "signbit", args, T.TY_INT)


@builtin_codegen("signbit")
def _gen_signbit(self, node) -> None:
    self.gen_expr(node.args[0])  # double in %xmm0
    self.emit("movq %xmm0, %rax")
    self.emit("shr $63, %rax")
