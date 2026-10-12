"""128x128 -> 128 wrapped multiply with exact overflow detection, for
__builtin_mul_overflow[_p] on __int128 operands (see overflow128.py).

Entry: a in %rdx:%rax, b in %rsi:%rdi (already in the operand domain).
Exit:  wrapped product in %rdx:%rax, domain overflow flag (0/1) in %ecx.
Clobbers %rcx, %rdx, %rsi, %rdi, %r9, %r10, %r11; preserves %r8.

Unsigned magnitude product P = A*B (mod 2^128) with overflow flag U when the
exact product is >= 2^128: U = (A1!=0 && B1!=0) | (A1*B0).hi!=0 |
(A0*B1).hi!=0 | carry out of H0 + (A1*B0).lo + (A0*B1).lo.

Signed: multiply |a| and |b|, remember neg = sign(a) ^ sign(b). The exact
signed product overflows iff |P| is >= 2^128, or P > 2^127 - 1 + neg. The
wrapped result is -P when neg, else P.
"""
from __future__ import annotations


def _emit_unsigned_core(emit) -> None:
    emit("mov %rax, %r9")            # A0
    emit("mov %rdx, %r10")           # A1
    emit("mov %rdi, %r11")           # B0
    emit("xor %ecx, %ecx")
    emit("test %r10, %r10")
    emit("setne %cl")
    emit("test %rsi, %rsi")
    emit("setne %dl")
    emit("and %dl, %cl")             # U = A1 && B1
    emit("mov %r9, %rax")
    emit("mul %r11")                 # A0*B0 -> rdx:rax
    emit("push %rax")                # low word of the result
    emit("mov %rdx, %rdi")           # H0
    emit("mov %r10, %rax")
    emit("mul %r11")                 # A1*B0
    emit("mov %rax, %r11")           # X1 low
    emit("test %rdx, %rdx")
    emit("setne %dl")
    emit("or %dl, %cl")
    emit("mov %r9, %rax")
    emit("mul %rsi")                 # A0*B1
    emit("test %rdx, %rdx")
    emit("setne %dl")
    emit("or %dl, %cl")
    emit("add %r11, %rdi")
    emit("setc %r11b")
    emit("movzbl %r11b, %r11d")
    emit("add %rax, %rdi")
    emit("setc %al")
    emit("movzbl %al, %eax")
    emit("or %r11d, %eax")
    emit("or %eax, %ecx")
    emit("pop %rax")
    emit("mov %rdi, %rdx")


def emit_mul128(self, signed: bool) -> None:
    emit = self.emit
    if not signed:
        _emit_unsigned_core(emit)
        return
    # neg = sign(a) ^ sign(b), kept on the stack across the core.
    emit("mov %rdx, %r9")
    emit("shr $63, %r9")
    emit("mov %rsi, %r10")
    emit("shr $63, %r10")
    emit("xor %r10, %r9")
    emit("push %r9")
    # |a| and |b| as conditional 128-bit negations.
    emit("mov %rdx, %r10")
    emit("sar $63, %r10")
    emit("xor %r10, %rax")
    emit("xor %r10, %rdx")
    emit("sub %r10, %rax")
    emit("sbb %r10, %rdx")
    emit("mov %rsi, %r11")
    emit("sar $63, %r11")
    emit("xor %r11, %rdi")
    emit("xor %r11, %rsi")
    emit("sub %r11, %rdi")
    emit("sbb %r11, %rsi")
    _emit_unsigned_core(emit)
    emit("pop %r9")                  # neg
    # ex = (P == 2^127); overflow if P's top bit is set, unless neg && ex.
    emit("mov %rdx, %r10")
    emit("btc $63, %r10")
    emit("or %rax, %r10")
    emit("sete %r10b")
    emit("movzbl %r10b, %r10d")
    emit("and %r9, %r10")
    emit("xor $1, %r10")
    emit("mov %rdx, %r11")
    emit("shr $63, %r11")
    emit("and %r11, %r10")
    emit("or %r10d, %ecx")
    # result = neg ? -P : P
    emit("mov %r9, %r10")
    emit("neg %r10")
    emit("xor %r10, %rax")
    emit("xor %r10, %rdx")
    emit("add %r9, %rax")
    emit("adc $0, %rdx")
