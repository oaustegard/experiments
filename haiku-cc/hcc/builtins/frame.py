"""__builtin_return_address(n) / __builtin_frame_address(n): walk the %rbp chain."""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.builtins import builtin_codegen, builtin_parser
from hcc.sema.constexpr import try_eval_const


def _frame_level(self, tok, name):
    """__builtin_{return,frame}_address(n): n must be a non-negative integer
    constant. The level is passed as an IntLit; codegen walks the %rbp chain."""
    args = self._args(1)
    level = try_eval_const(args[0])
    if level is None or level < 0:
        self.error(f"'__builtin_{name}' needs a non-negative integer constant")
    return self._call(tok, name, [N.IntLit(tok.loc, level, ty=T.TY_INT)],
                      T.pointer_to(T.TY_VOID))


@builtin_parser("__builtin_return_address")
def _parse_return_address(self, tok):
    return _frame_level(self, tok, "return_address")


@builtin_parser("__builtin_frame_address")
def _parse_frame_address(self, tok):
    return _frame_level(self, tok, "frame_address")


def _frame_walk(self, node) -> None:
    """%rax = frame pointer of level args[0]: every hcc function sets %rbp
    as its frame pointer and saves the caller's %rbp at 0(%rbp)."""
    self.emit("mov %rbp, %rax")
    for _ in range(node.args[0].value):
        self.emit("mov (%rax), %rax")


@builtin_codegen("frame_address")
def _gen_frame_address(self, node) -> None:
    _frame_walk(self, node)


@builtin_codegen("return_address")
def _gen_return_address(self, node) -> None:
    _frame_walk(self, node)
    self.emit("mov 8(%rax), %rax")
