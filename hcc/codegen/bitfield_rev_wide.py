"""Reverse-order (scalar_storage_order "big-endian") bitfields whose byte span
is wider than 8 bytes, e.g. a 124-bit `__uint128_t` field.

The span is a big-endian integer B of n (<= 16) bytes at (%rsi). It is kept in
%r11:%r10 (high:low word) while being manipulated. The field is
B >> bo, masked to bit_width. The field value travels as a 128-bit value in %rdx:%rax.
Register use: %rsi span address, %r8/%r9 the shifted value, %r10/%r11 span
words, %rcx mask scratch, %rdx byte scratch.
"""
from __future__ import annotations

from hcc.codegen.bitfield_wide import _M64, _movabs_and, _shl128, _shr128, field_to_rdx_rax


def _load_be_span(cg, n: int) -> None:
    """%r11:%r10 = big-endian integer of the n bytes at (%rsi)
    (%r10 low word, %r11 high word, as in bitfield_wide)."""
    cg.emit("xor %r10d, %r10d")
    cg.emit("xor %r11d, %r11d")
    for k in range(n):
        cg.emit("shld $8, %r10, %r11")
        cg.emit("shl $8, %r10")
        cg.emit(f"movzbl {k}(%rsi), %edx")
        cg.emit("or %rdx, %r10")


def _store_be_span(cg, n: int) -> None:
    """Store the n low-order bytes of %r11:%r10 big-endian at (%rsi)."""
    for k in range(n):
        s = 8 * (n - 1 - k)
        if s < 64:
            cg.emit("mov %r10, %rdx")
            if s:
                cg.emit(f"shr ${s}, %rdx")
        else:
            cg.emit("mov %r11, %rdx")
            if s > 64:
                cg.emit(f"shr ${s - 64}, %rdx")
        cg.emit(f"mov %dl, {k}(%rsi)")


def load_rev_wide(cg, m) -> None:
    """Address of the member in %rax -> field value in %rdx:%rax (128 bits)."""
    w, bo, n = m.bit_width, m.bit_offset, m.rev_bytes
    cg.emit("mov %rax, %rsi")
    _load_be_span(cg, n)
    _shr128(cg, bo)
    field_to_rdx_rax(cg, w, m.ty.is_unsigned)


def assign_rev_wide(cg, node) -> None:
    """`lhs = rhs` for a wide reverse-order bitfield. Called after the lhs
    address was pushed and the rhs value is in %rax; leaves the assigned value
    (as the field type) in %rax."""
    m = node.lhs.member
    w, bo, n = m.bit_width, m.bit_offset, m.rev_bytes
    field = ((1 << w) - 1) << bo              # field bits inside the span
    cg.emit("mov %rax, %r8")                  # value low word
    cg.emit("mov %rdx, %r9")                  # value high word
    _movabs_and(cg, (1 << min(w, 64)) - 1, "%r8")
    _movabs_and(cg, (1 << max(0, min(w - 64, 64))) - 1, "%r9")
    _shl128(cg, bo)
    cg.pop("%rsi")                            # span address
    _load_be_span(cg, n)
    _movabs_and(cg, ~(field & _M64) & _M64, "%r10")
    _movabs_and(cg, ~(field >> 64) & _M64, "%r11")
    cg.emit("or %r8, %r10")
    cg.emit("or %r9, %r11")
    _store_be_span(cg, n)
    cg.emit("mov %rsi, %rax")                 # the assignment yields the stored field
    load_rev_wide(cg, m)
