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
        """Allocate %rax bytes (rounded to 16) on the stack; result in %rax.

        The stack machine may have `self.depth` pushed temporaries live at
        (%rsp), so the logical base is rsp + depth*8. The block is carved
        just below that base (16-aligned, gcc-like). The temporaries are
        copied to just below the block (ascending, so overlap is safe since
        the destination is always lower), and the block's bottom becomes the
        new depth-0 base, so later pops never reach the block."""
        d = self.depth
        self.emit("add $15, %rax")
        self.emit("and $-16, %rax")
        self.emit("mov %rsp, %r11")
        self.emit(f"add ${d * 8}, %r11")       # r11 = logical base
        self.emit("sub %rax, %r11")
        self.emit("and $-16, %r11")            # r11 = block address (new base)
        for i in range(d):
            self.emit(f"mov {i * 8}(%rsp), %r10")
            self.emit(f"mov %r10, {i * 8 - d * 8}(%r11)")
        self.emit("mov %r11, %rax")
        self.emit(f"lea {-d * 8}(%r11), %rsp")
