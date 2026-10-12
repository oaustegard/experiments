"""Compile-time type queries: __builtin_types_compatible_p, __builtin_classify_type,
__builtin_choose_expr."""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.builtins import builtin_parser
from hcc.sema.compat import is_compatible


@builtin_parser("__builtin_classify_type")
def _parse_classify_type(self, tok):
    args = self._args(1)
    t = args[0].ty
    v = 1 if T.is_integer(t) else 8 if T.is_flonum(t) else 5 if t.base is not None else 12
    return N.IntLit(tok.loc, v, ty=T.TY_INT)


@builtin_parser("__builtin_types_compatible_p")
def _parse_types_compatible_p(self, tok):
    a = self.typename().unqual()
    self.expect(",")
    b = self.typename().unqual()
    return N.IntLit(tok.loc, int(is_compatible(a, b)), ty=T.TY_INT)


@builtin_parser("__builtin_choose_expr")
def _parse_choose_expr(self, tok):
    c = self.const_expr()
    self.expect(",")
    a = self.assign()
    self.expect(",")
    b = self.assign()
    return a if c else b
