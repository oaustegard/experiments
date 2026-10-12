"""__builtin_{add,sub}_overflow[_p] on __int128 operands (operands typed by
parser/overflow_wide.py: both 128-bit, in one domain).

The wrapped 128-bit result is computed in %rdx:%rax. The domain's overflow flag
(OF for signed, CF for unsigned) says the exact result left the 128-bit domain.
The result fits the target type T iff there is no flag and, for a 128-bit T,
(unsigned T in a signed domain) the value is non-negative; for a narrower T the
value must equal its own low bits sign- or zero-extended. The stored value is
the wrapped result truncated to T, as C requires; the builtin yields 1 iff it
does not fit.
"""
from __future__ import annotations

from hcc.codegen.overflow_narrow import narrow_fit_check
from hcc.codegen.overflow128_mul import emit_mul128


class Overflow128Mixin:

    def overflow128(self, node, op: str, store: bool) -> None:
        T = node.type_arg
        dom_signed = not node.args[0].ty.is_unsigned
        if store:
            self.gen_expr(node.args[2])
            self.push()                  # result pointer
        self.gen_expr(node.args[1])
        self.push128()
        self.gen_expr(node.args[0])
        self.pop128_args()               # b -> %rsi:%rdi, a -> %rdx:%rax
        if store:
            self.pop("%r8")
        if op == "mul":
            emit_mul128(self, dom_signed)    # %rdx:%rax wrapped, %ecx = flag
        else:
            if op == "add":
                self.emit("add %rdi, %rax")
                self.emit("adc %rsi, %rdx")
            else:
                self.emit("sub %rdi, %rax")
                self.emit("sbb %rsi, %rdx")
            self.emit("seto %cl" if dom_signed else "setc %cl")
            self.emit("movzbl %cl, %ecx")    # %ecx = domain overflow flag
        if T.size == 16:
            self.emit("xor %r11d, %r11d")
            if T.is_unsigned and dom_signed:
                self.emit("mov %rdx, %r10")
                self.emit("shr $63, %r10")
                self.emit("or %r10d, %r11d")
            self.emit("or %ecx, %r11d")
            if store:
                self.emit("mov %rax, (%r8)")
                self.emit("mov %rdx, 8(%r8)")
        else:
            narrow_fit_check(self, T, store)     # %r11d := not fit, includes %ecx
        self.emit("movzbl %r11b, %eax")
