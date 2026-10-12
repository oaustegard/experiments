"""Store-and-fit check for overflow builtins whose result type is 1-8 bytes.

narrow_fit_check(cg, T, store): with the wrapped 128-bit result in %rdx:%rax and
the domain overflow flag in %ecx, stores the low bytes of %rax to (%r8) when
`store`, and sets %r11d to 1 iff the value does not fit T (or %ecx is set).
Clobbers %r9, %r10, %r11 and the flags.
"""
from __future__ import annotations


def narrow_fit_check(cg, T, store: bool) -> None:
    size = T.size if T.size in (1, 2, 4, 8) else 8
    signed = not T.is_unsigned
    if size == 1:
        if store:
            cg.emit("mov %al, (%r8)")
        cg.emit("movsbq %al, %r9" if signed else "movzbq %al, %r9")
    elif size == 2:
        if store:
            cg.emit("mov %ax, (%r8)")
        cg.emit("movswq %ax, %r9" if signed else "movzwq %ax, %r9")
    elif size == 4:
        if store:
            cg.emit("mov %eax, (%r8)")
        cg.emit("movslq %eax, %r9" if signed else "movl %eax, %r9d")
    else:
        if store:
            cg.emit("mov %rax, (%r8)")
        cg.emit("mov %rax, %r9")
    cg.emit("xor %r11d, %r11d")
    cg.emit("cmp %r9, %rax")
    cg.emit("setne %r11b")
    if signed:
        cg.emit("mov %r9, %r10")
        cg.emit("sar $63, %r10")
    else:
        cg.emit("xor %r10d, %r10d")
    cg.emit("cmp %r10, %rdx")
    cg.emit("setne %r10b")
    cg.emit("or %r10b, %r11b")
    cg.emit("or %ecx, %r11d")
