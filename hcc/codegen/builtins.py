"""Lowering of nodes.BuiltinCall: one `builtin_<name>` method per builtin
(name without the `__builtin_` prefix). va_* live in varargs.py.
To add a builtin: parse it in parser/builtins.py, implement builtin_<name> here.
"""
from __future__ import annotations

from hcc import diag


class BuiltinMixin:

    def expr_BuiltinCall(self, node) -> None:
        fn = getattr(self, "builtin_" + node.name, None)
        if fn is None:
            diag.error(node, f"codegen: unsupported builtin '__builtin_{node.name}'", "codegen")
        fn(node)

    def builtin_trap(self, node) -> None:
        self.emit("ud2")

    def builtin_bswap16(self, node) -> None:
        self.gen_expr(node.args[0])
        self.emit("rol $8, %ax")
        self.emit("movzwl %ax, %eax")

    def builtin_bswap32(self, node) -> None:
        self.gen_expr(node.args[0])
        self.emit("bswap %eax")

    def builtin_bswap64(self, node) -> None:
        self.gen_expr(node.args[0])
        self.emit("bswap %rax")

    def builtin_alloca(self, node) -> None:
        self.gen_expr(node.args[0])
        self._alloca_rax()

    def builtin_alloca_call(self, node) -> None:
        """A plain call to `alloca(n)` (glibc's alloca.h maps it here)."""
        self.gen_expr(node.args[0])
        self._alloca_rax()

    def _alloca_rax(self) -> None:
        # NOTE: assumes no pushed temporaries are live (depth == 0) — TODO
        self.emit("add $15, %rax")
        self.emit("and $-16, %rax")
        self.emit("sub %rax, %rsp")
        self.emit("mov %rsp, %rax")
