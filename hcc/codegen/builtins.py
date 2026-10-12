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

    def _frame_walk(self, node) -> None:
        """%rax = frame pointer of level args[0]: every hcc function sets %rbp
        as its frame pointer and saves the caller's %rbp at 0(%rbp)."""
        self.emit("mov %rbp, %rax")
        for _ in range(node.args[0].value):
            self.emit("mov (%rax), %rax")

    def builtin_frame_address(self, node) -> None:
        self._frame_walk(node)

    def builtin_return_address(self, node) -> None:
        self._frame_walk(node)
        self.emit("mov 8(%rax), %rax")

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

    def builtin_signbit(self, node) -> None:
        self.gen_expr(node.args[0])  # double in %xmm0
        self.emit("movq %xmm0, %rax")
        self.emit("shr $63, %rax")

    # Bit-counting builtins. Results are fully defined even for a zero
    # argument (GCC leaves clz/ctz of 0 undefined): clz/ctz give the width,
    # ffs gives 0, popcount/parity give 0.
    def builtin_clz32(self, node) -> None:
        self._bit_arg32(node)
        self.emit("mov $-1, %edx")
        self.emit("bsr %eax, %eax")
        self.emit("cmovz %edx, %eax")
        self.emit("neg %eax")
        self.emit("add $31, %eax")

    def builtin_clz64(self, node) -> None:
        self.gen_expr(node.args[0])
        self.emit("mov $-1, %edx")
        self.emit("bsr %rax, %rax")
        self.emit("cmovz %edx, %eax")
        self.emit("neg %eax")
        self.emit("add $63, %eax")

    def builtin_ctz32(self, node) -> None:
        self._bit_arg32(node)
        self.emit("mov $32, %edx")
        self.emit("bsf %eax, %eax")
        self.emit("cmovz %edx, %eax")

    def builtin_ctz64(self, node) -> None:
        self.gen_expr(node.args[0])
        self.emit("mov $64, %edx")
        self.emit("bsf %rax, %rax")
        self.emit("cmovz %edx, %eax")

    def builtin_ffs32(self, node) -> None:
        self._bit_arg32(node)
        self.emit("mov $-1, %edx")
        self.emit("bsf %eax, %eax")
        self.emit("cmovz %edx, %eax")
        self.emit("add $1, %eax")

    def builtin_ffs64(self, node) -> None:
        self.gen_expr(node.args[0])
        self.emit("mov $-1, %edx")
        self.emit("bsf %rax, %rax")
        self.emit("cmovz %edx, %eax")
        self.emit("add $1, %eax")

    # clrsb(x): number of redundant sign bits below the sign bit, i.e.
    # clz(x ^ (x >> (w-1))) - 1. Defined for 0 and -1 (gives w-1).
    def builtin_clrsb32(self, node) -> None:
        self.gen_expr(node.args[0])
        self.emit("mov %eax, %edx")
        self.emit("sar $31, %edx")
        self.emit("xor %edx, %eax")
        self.emit("mov $-1, %edx")
        self.emit("bsr %eax, %eax")
        self.emit("cmovz %edx, %eax")
        self.emit("neg %eax")
        self.emit("add $30, %eax")

    def builtin_clrsb64(self, node) -> None:
        self.gen_expr(node.args[0])
        self.emit("mov %rax, %rdx")
        self.emit("sar $63, %rdx")
        self.emit("xor %rdx, %rax")
        self.emit("mov $-1, %edx")
        self.emit("bsr %rax, %rax")
        self.emit("cmovz %edx, %eax")
        self.emit("neg %eax")
        self.emit("add $62, %eax")

    def builtin_popcount32(self, node) -> None:
        self._bit_arg32(node)
        self._popcount32()

    def builtin_popcount64(self, node) -> None:
        self.gen_expr(node.args[0])
        self._popcount64()

    def builtin_parity32(self, node) -> None:
        self._bit_arg32(node)
        self._popcount32()
        self.emit("and $1, %eax")

    def builtin_parity64(self, node) -> None:
        self.gen_expr(node.args[0])
        self._popcount64()
        self.emit("and $1, %eax")

    def _bit_arg32(self, node) -> None:
        self.gen_expr(node.args[0])
        self.emit("mov %eax, %eax")  # clear any upper garbage

    def _popcount32(self) -> None:
        """SWAR population count of %eax (clobbers %edx)."""
        self.emit("mov %eax, %edx")
        self.emit("shr $1, %edx")
        self.emit("and $0x55555555, %edx")
        self.emit("sub %edx, %eax")
        self.emit("mov %eax, %edx")
        self.emit("and $0x33333333, %eax")
        self.emit("shr $2, %edx")
        self.emit("and $0x33333333, %edx")
        self.emit("add %edx, %eax")
        self.emit("mov %eax, %edx")
        self.emit("shr $4, %edx")
        self.emit("add %edx, %eax")
        self.emit("and $0x0f0f0f0f, %eax")
        self.emit("imul $0x01010101, %eax, %eax")
        self.emit("shr $24, %eax")

    def _popcount64(self) -> None:
        """SWAR population count of %rax (clobbers %rdx, %rcx)."""
        self.emit("mov %rax, %rdx")
        self.emit("shr $1, %rdx")
        self.emit("movabs $0x5555555555555555, %rcx")
        self.emit("and %rcx, %rdx")
        self.emit("sub %rdx, %rax")
        self.emit("movabs $0x3333333333333333, %rcx")
        self.emit("mov %rax, %rdx")
        self.emit("and %rcx, %rax")
        self.emit("shr $2, %rdx")
        self.emit("and %rcx, %rdx")
        self.emit("add %rdx, %rax")
        self.emit("mov %rax, %rdx")
        self.emit("shr $4, %rdx")
        self.emit("add %rdx, %rax")
        self.emit("movabs $0x0f0f0f0f0f0f0f0f, %rcx")
        self.emit("and %rcx, %rax")
        self.emit("movabs $0x0101010101010101, %rcx")
        self.emit("imul %rcx, %rax")
        self.emit("shr $56, %rax")

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
