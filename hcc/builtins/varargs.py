"""__builtin_va_* (stdarg.h). Lowered by codegen/varargs.py (builtin_va_start/va_copy/va_arg);
va_arg_pack placeholders are resolved by parser/va_pack.py."""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc import typesys as T
from hcc.builtins import builtin_parser
from hcc.sema.incomplete import is_incomplete_object


@builtin_parser("__builtin_va_start")
def _parse_va_start(self, tok):
    args = self._args()
    return self._call(tok, "va_start", args[:1], T.TY_VOID)


@builtin_parser("__builtin_va_arg_pack")
def _parse_va_arg_pack(self, tok):
    """Placeholder resolved by parser/va_pack.py (extra args of the inlined call)."""
    return self._call(tok, "va_arg_pack", [], T.TY_INT)


@builtin_parser("__builtin_va_arg_pack_len")
def _parse_va_arg_pack_len(self, tok):
    return self._call(tok, "va_arg_pack_len", [], T.TY_INT)


@builtin_parser("__builtin_va_end")
def _parse_va_end(self, tok):
    self._args()
    return N.Cast(tok.loc, N.IntLit(tok.loc, 0, ty=T.TY_INT), ty=T.TY_VOID)


@builtin_parser("__builtin_va_copy")
def _parse_va_copy(self, tok):
    return self._call(tok, "va_copy", self._args(2), T.TY_VOID)


@builtin_parser("__builtin_va_arg")
def _parse_va_arg(self, tok):
    ap = self.typed(self.assign())
    self.expect(",")
    ty = self.typename()
    # the second argument must be a complete object type (C11 7.16.1.1p2)
    if ty.kind == T.TypeKind.VOID or is_incomplete_object(ty):
        diag.error(tok.loc, f"second argument to 'va_arg' is of incomplete type '{ty}'", "sema")
    # va_arg yields an rvalue; an array type can never be produced by it
    if ty.kind in (T.TypeKind.ARRAY, T.TypeKind.VLA):
        diag.error(tok.loc, "invalid use of non-lvalue array")
    if ty.kind == T.TypeKind.FUNC:
        diag.error(tok.loc, "second argument to 'va_arg' is a function type")
    # a va_list may be consumed by any function (not only variadic ones,
    # e.g. `void f(va_list ap)`), so make sure the split-struct scratch exists
    fn = getattr(self, "fn", None)
    if fn is not None and fn.va_tmp is None:
        fn.va_tmp = self.new_temp(T.array_of(T.TY_CHAR, 16))
    # `va_arg(ap, typeof(int (*)[++i]))`: the VLA bounds are evaluated first
    ty, pre = self.vla_bind(ty, tok.loc)
    return self.vla_prefix(pre, self._call(tok, "va_arg", [ap], ty, ty), tok.loc)
