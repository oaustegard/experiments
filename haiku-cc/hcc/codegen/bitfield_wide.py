"""Bitfields whose declared type is 16 bytes (__int128 / __uint128_t).

The member's storage is two 64-bit words at its address (`%rax` in loads,
`%rdi` in stores). `m.bit_offset` is the field's LSB position relative to
that address. All shift counts and masks are compile-time constants, so
the 128-bit arithmetic is emitted with shrd/shld on %r8-%r11 and never needs
a 128-bit immediate. The field value travels as a 128-bit value in %rdx:%rax.
Register use: %rcx mask scratch, %r8/%r9 value words, %r10/%r11 storage
words, %rdi the address, %rax the value.
"""
from __future__ import annotations

_M64 = (1 << 64) - 1


def _shr128(cg, n: int) -> None:
    """%r11:%r10 >>= n (0 <= n < 128), logical."""
    if n == 0:
        return
    if n < 64:
        cg.emit(f"shrd ${n}, %r11, %r10")
        cg.emit(f"shr ${n}, %r11")
    else:
        cg.emit("mov %r11, %r10")
        if n > 64:
            cg.emit(f"shr ${n - 64}, %r10")
        cg.emit("xor %r11d, %r11d")


def _shl128(cg, n: int) -> None:
    """%r9:%r8 <<= n (0 <= n < 128)."""
    if n == 0:
        return
    if n < 64:
        cg.emit(f"shld ${n}, %r8, %r9")
        cg.emit(f"shl ${n}, %r8")
    else:
        cg.emit("mov %r8, %r9")
        if n > 64:
            cg.emit(f"shl ${n - 64}, %r9")
        cg.emit("xor %r8d, %r8d")


def _movabs_and(cg, imm: int, reg: str) -> None:
    cg.emit(f"movabs ${imm}, %rcx")
    cg.emit(f"and %rcx, {reg}")


def field_to_rdx_rax(cg, w: int, unsigned: bool) -> None:
    """%r11:%r10 holds the field shifted down to bit 0 (bits above w are junk).
    Leaves the field extended to 128 bits in %rdx:%rax."""
    if w < 64:
        cg.emit(f"shl ${64 - w}, %r10")
        cg.emit(f"{'shr' if unsigned else 'sar'} ${64 - w}, %r10")
        _extend_high(cg, unsigned)
    elif w == 64:
        _extend_high(cg, unsigned)
    else:
        if w < 128:
            cg.emit(f"shl ${128 - w}, %r11")
            cg.emit(f"{'shr' if unsigned else 'sar'} ${128 - w}, %r11")
        cg.emit("mov %r11, %rdx")
    cg.emit("mov %r10, %rax")


def load_wide_bitfield(cg, m) -> None:
    """%rdx:%rax = the field at (%rax) (address of the member), extended to 128 bits."""
    cg.emit("mov (%rax), %r10")
    cg.emit("mov 8(%rax), %r11")
    _shr128(cg, m.bit_offset)
    field_to_rdx_rax(cg, m.bit_width, m.ty.is_unsigned)


def _extend_high(cg, unsigned: bool) -> None:
    """%rdx = the high word of the 64-bit value in %r10 (sign or zero extension)."""
    if unsigned:
        cg.emit("xor %edx, %edx")
    else:
        cg.emit("mov %r10, %rdx")
        cg.emit("sar $63, %rdx")


def assign_wide_bitfield(cg, node) -> None:
    """`lhs = rhs` for a wide bitfield. %rax holds the rhs value and the lhs
    member address is on top of the stack. Leaves the assigned value (as the
    field type) in %rax."""
    m = node.lhs.member
    bo, w = m.bit_offset, m.bit_width
    field = ((1 << w) - 1) << bo              # 128-bit mask of the field
    cg.emit("mov %rax, %r8")                  # value low word
    cg.emit("mov %rdx, %r9")                  # value high word
    _movabs_and(cg, (1 << min(w, 64)) - 1, "%r8")          # keep w bits
    _movabs_and(cg, (1 << max(0, min(w - 64, 64))) - 1, "%r9")
    _shl128(cg, bo)
    cg.pop("%rdi")
    cg.emit("mov (%rdi), %r10")
    cg.emit("mov 8(%rdi), %r11")
    _movabs_and(cg, ~(field & _M64) & _M64, "%r10")
    _movabs_and(cg, ~(field >> 64) & _M64, "%r11")
    cg.emit("or %r8, %r10")
    cg.emit("or %r9, %r11")
    cg.emit("mov %r10, (%rdi)")
    cg.emit("mov %r11, 8(%rdi)")
    cg.emit("mov %rdi, %rax")                 # the assignment yields the stored field
    load_wide_bitfield(cg, m)
