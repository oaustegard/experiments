"""__builtin_{add,sub,mul}_overflow(a, b, &res).

The parser widens each operand to LONG (signed type) or ULONG (unsigned type),
so every operand value is exact in 64 bits. The operation is computed exactly
in 128 bits in %rdx:%rax; the result is stored truncated to the type of *res
and the builtin yields 1 iff the exact result does not fit that type.
"""
from __future__ import annotations


class OverflowMixin:

    def builtin_add_overflow(self, node) -> None:
        self._overflow_op(node, "add")

    def builtin_sub_overflow(self, node) -> None:
        self._overflow_op(node, "sub")

    def builtin_mul_overflow(self, node) -> None:
        self._overflow_op(node, "mul")

    def builtin_add_overflow_p(self, node) -> None:
        self._overflow_p(node, "add")

    def builtin_sub_overflow_p(self, node) -> None:
        self._overflow_p(node, "sub")

    def builtin_mul_overflow_p(self, node) -> None:
        """Like mul_overflow but only reports whether the product fits the type
        of the third argument; nothing is stored."""
        self._overflow_p(node, "mul")

    def _overflow_p(self, node, op: str) -> None:
        sa = not node.args[0].ty.is_unsigned
        sb = not node.args[1].ty.is_unsigned
        for arg in node.args[:2]:
            self.gen_expr(arg)
            self.push()
        self.pop("%rcx")     # b
        self.pop("%rsi")     # a
        if op == "mul":
            self._exact_mul(sa, sb)
        else:
            self._exact_addsub(op, sa, sb)
        self._store_and_check(node.type_arg, store=False)

    def _overflow_op(self, node, op: str) -> None:
        sa = not node.args[0].ty.is_unsigned
        sb = not node.args[1].ty.is_unsigned
        T = node.type_arg
        for arg in node.args:
            self.gen_expr(arg)
            self.push()
        self.pop("%r8")      # result pointer
        self.pop("%rcx")     # b
        self.pop("%rsi")     # a
        if op == "mul":
            self._exact_mul(sa, sb)
        else:
            self._exact_addsub(op, sa, sb)
        self._store_and_check(T)

    def _exact_addsub(self, op: str, sa: bool, sb: bool) -> None:
        """Signed 128-bit exact sum/difference of two 64-bit operands -> %r9:%rax."""
        self.emit("mov %rsi, %rax")
        if sa:
            self.emit("mov %rsi, %r9")
            self.emit("sar $63, %r9")
        else:
            self.emit("xor %r9d, %r9d")
        self.emit("mov %rcx, %r10")
        if sb:
            self.emit("mov %rcx, %r11")
            self.emit("sar $63, %r11")
        else:
            self.emit("xor %r11d, %r11d")
        if op == "add":
            self.emit("add %r10, %rax")
            self.emit("adc %r11, %r9")
        else:
            self.emit("sub %r10, %rax")
            self.emit("sbb %r11, %r9")
        self.emit("mov %r9, %rdx")

    def _exact_mul(self, sa: bool, sb: bool) -> None:
        """Exact 128-bit product of two 64-bit operands -> %rdx:%rax."""
        if not sa and not sb:
            self.emit("mov %rsi, %rax")
            self.emit("mulq %rcx")
            return
        if sa and sb:
            self.emit("mov %rsi, %rax")
            self.emit("imulq %rcx")
            return
        # Mixed: the unsigned operand u equals (u as signed) + 2^64 when its
        # sign bit is set; that extra term adds the signed operand to %rdx.
        s, u = ("%rsi", "%rcx") if sa else ("%rcx", "%rsi")
        done = self.new_label()
        self.emit(f"mov {s}, %rax")
        self.emit(f"imulq {u}")
        self.emit(f"test {u}, {u}")
        self.emit(f"jns {done}")
        self.emit(f"add {s}, %rdx")
        self.label(done)

    def _store_and_check(self, T, store: bool = True) -> None:
        """Store low bits of %rax to (%r8) (if store); %eax = 1 iff %rdx:%rax does not fit T."""
        size = T.size if T.size in (1, 2, 4, 8) else 8
        signed = not T.is_unsigned
        if size == 1:
            if store:
                self.emit("mov %al, (%r8)")
            self.emit("movsbq %al, %r9" if signed else "movzbq %al, %r9")
        elif size == 2:
            if store:
                self.emit("mov %ax, (%r8)")
            self.emit("movswq %ax, %r9" if signed else "movzwq %ax, %r9")
        elif size == 4:
            if store:
                self.emit("mov %eax, (%r8)")
            self.emit("movslq %eax, %r9" if signed else "movl %eax, %r9d")
        else:
            if store:
                self.emit("mov %rax, (%r8)")
            self.emit("mov %rax, %r9")
        # The value fits iff its sign/zero extension equals the exact result.
        self.emit("xor %r11d, %r11d")
        self.emit("cmp %r9, %rax")
        self.emit("setne %r11b")
        if signed:
            self.emit("mov %r9, %r10")
            self.emit("sar $63, %r10")
        else:
            self.emit("xor %r10d, %r10d")
        self.emit("cmp %r10, %rdx")
        self.emit("setne %r10b")
        self.emit("or %r10b, %r11b")
        self.emit("movzbl %r11b, %eax")
