"""__builtin_{add,sub,mul}_overflow[_p]. Lowered by codegen/overflow.py."""
from __future__ import annotations

from hcc import typesys as T
from hcc.builtins import builtin_parser
from hcc.parser.overflow_wide import wide_overflow_operands
from hcc.sema.conversions import new_cast


@builtin_parser("__builtin_add_overflow", "__builtin_sub_overflow", "__builtin_mul_overflow")
def _parse_overflow(self, tok):
    """__builtin_{add,sub,mul}_overflow(a, b, &res): operands are widened to
    exact 64-bit LONG/ULONG (by their own signedness); codegen does the check."""
    name = tok.text[len("__builtin_"):-len("_overflow")]
    args = self._args(3)
    res = args[2]
    if res.ty.kind != T.TypeKind.PTR or res.ty.base is None:
        self.error(f"'__builtin_{name}_overflow' expects a pointer as third argument")
    ops = wide_overflow_operands(self, name, args[:2], res.ty.base)
    if ops is None:
        ops = [new_cast(a, T.TY_ULONG if a.ty.is_unsigned else T.TY_LONG) for a in args[:2]]
    return self._call(tok, name + "_overflow", ops + [res], T.TY_INT, res.ty.base)


@builtin_parser("__builtin_add_overflow_p", "__builtin_sub_overflow_p", "__builtin_mul_overflow_p")
def _parse_overflow_p(self, tok):
    """__builtin_{add,sub,mul}_overflow_p(a, b, c): 1 iff a OP b does not fit
    the type of c. Only the type of c is used (its value is never read or stored)."""
    name = tok.text[len("__builtin_"):-len("_overflow_p")]
    args = self._args(3)
    ops = wide_overflow_operands(self, name, args[:2], args[2].ty)
    if ops is None:
        ops = [new_cast(a, T.TY_ULONG if a.ty.is_unsigned else T.TY_LONG) for a in args[:2]]
    return self._call(tok, name + "_overflow_p", ops, T.TY_INT, args[2].ty)
