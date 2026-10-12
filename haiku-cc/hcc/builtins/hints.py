"""Builtins that only carry a hint or a compile-time answer: __builtin_expect,
__builtin_constant_p, __builtin_prefetch, __builtin_assume_aligned,
__builtin_assoc_barrier, __builtin_unreachable, __builtin_trap."""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.builtins import builtin_codegen, builtin_fold, builtin_parser
from hcc.sema.constexpr import builtin_constant_p
from hcc.sema.conversions import new_cast


@builtin_parser("__builtin_prefetch")
def _parse_prefetch(self, tok):
    """__builtin_prefetch(addr[, rw[, locality]]) is only a cache hint: it
    evaluates its arguments (for side effects) and yields void."""
    args = self._args()
    if not 1 <= len(args) <= 3:
        self.error("builtin expects 1 to 3 arguments")
    node = N.Cast(tok.loc, N.IntLit(tok.loc, 0, ty=T.TY_INT), ty=T.TY_VOID)
    for a in reversed(args):
        node = N.Comma(tok.loc, N.Cast(tok.loc, a, ty=T.TY_VOID), node, ty=T.TY_VOID)
    return node


@builtin_parser("__builtin_assoc_barrier")
def _parse_assoc_barrier(self, tok):
    # Identity on the value: it only stops reassociation of floating expressions.
    return self._args(1)[0]


@builtin_parser("__builtin_assume_aligned")
def _parse_assume_aligned(self, tok):
    """__builtin_assume_aligned(p, align[, offset]) yields p unchanged (the
    alignment is only an optimisation hint); the other arguments are evaluated
    for their side effects, as in __builtin_prefetch."""
    args = self._args()
    if not 2 <= len(args) <= 3:
        self.error("builtin expects 2 or 3 arguments")
    node = args[0]
    for a in reversed(args[1:]):
        node = N.Comma(tok.loc, N.Cast(tok.loc, a, ty=T.TY_VOID), node, ty=node.ty)
    return node


@builtin_parser("__builtin_expect")
def _parse_expect(self, tok):
    args = self._args(2)
    # The second argument (the expected value) is still evaluated for its
    # side effects; the result is the first argument.
    seq = self.typed(N.Comma(tok.loc, args[1], args[0]))
    return new_cast(seq, T.TY_LONG)


@builtin_parser("__builtin_constant_p")
def _parse_constant_p(self, tok):
    args = self._args(1)
    return N.IntLit(tok.loc, builtin_constant_p(args[0]), ty=T.TY_INT)


@builtin_fold("__builtin_constant_p")
def _fold_constant_p(node, num):
    return builtin_constant_p(node.args[0])


@builtin_parser("__builtin_unreachable", "__builtin_trap")
def _parse_trap(self, tok):
    return self._call(tok, "trap", [], T.TY_VOID)


@builtin_codegen("trap")
def _gen_trap(self, node) -> None:
    self.emit("ud2")
