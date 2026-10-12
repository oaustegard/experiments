"""Inline integer abs() and float fabs() (BuiltinCalls made by parser/libc_builtins.py)."""
from __future__ import annotations

from hcc.builtins import builtin_codegen


@builtin_codegen("abs32")
def _gen_abs32(self, node) -> None:
    """Branch-free |x| for int: (x ^ m) - m with m = x >> 31."""
    self.gen_expr(node.args[0])
    self.emit("mov %eax, %edx")
    self.emit("sar $31, %edx")
    self.emit("xor %edx, %eax")
    self.emit("sub %edx, %eax")


@builtin_codegen("fabsf")
def _gen_fabsf(self, node) -> None:
    """|x| on a float in %xmm0: clear the sign bit (quiet/signaling bits kept)."""
    self.gen_expr(node.args[0])
    self.emit("movd %xmm0, %eax")
    self.emit("and $0x7fffffff, %eax")
    self.emit("movd %eax, %xmm0")


@builtin_codegen("abs64")
def _gen_abs64(self, node) -> None:
    self.gen_expr(node.args[0])
    self.emit("mov %rax, %rdx")
    self.emit("sar $63, %rdx")
    self.emit("xor %rdx, %rax")
    self.emit("sub %rdx, %rax")
