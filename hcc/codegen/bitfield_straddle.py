"""Scalar bitfields whose LSB offset plus width exceeds 64 bits.

This happens in packed structs, e.g. `uint64_t a : 64;` placed at bit 4: the
field spans the 8-byte word at its address plus one extra byte. The field is
then read from two storage words (`%rdx` low, `%rcx` high byte) and written
as a low word plus a single high byte. `m.bit_offset` is the LSB relative to
the member address (`%rax` in loads, `%rdi` in stores), and is in 1..7 here.
"""
from __future__ import annotations


def load_straddle_bitfield(cg, m) -> None:
    """%rax = address of the member. Leaves the sign/zero-extended field in %rax."""
    bo, w = m.bit_offset, m.bit_width
    cg.emit("mov (%rax), %rdx")
    cg.emit("movzbl 8(%rax), %ecx")
    cg.emit(f"shrd ${bo}, %rcx, %rdx")
    cg.emit("mov %rdx, %rax")
    if w < 64:
        cg.emit(f"shl ${64 - w}, %rax")
        cg.emit(f"{'shr' if m.ty.is_unsigned else 'sar'} ${64 - w}, %rax")


def assign_straddle_bitfield(cg, node) -> None:
    """`lhs = rhs` for a straddling bitfield. %rax holds the rhs value and the
    lhs member address is on top of the stack. Leaves the assigned value (as
    the field type) in %rax."""
    m = node.lhs.member
    bo, w = m.bit_offset, m.bit_width
    hb = bo + w - 64                          # bits of the field in byte 8
    cg.emit("mov %rax, %r8")                  # value
    cg.pop("%rdi")
    # Low storage word: keep the bits below the field, put value << bo above.
    cg.emit("mov (%rdi), %r9")
    cg.emit(f"movabs ${(1 << bo) - 1}, %rcx")
    cg.emit("and %rcx, %r9")
    cg.emit("mov %r8, %r10")
    cg.emit(f"shl ${bo}, %r10")
    cg.emit("or %r10, %r9")
    cg.emit("mov %r9, (%rdi)")
    # High byte: the field's top `hb` bits, keeping the other bits of the byte.
    cg.emit("movzbl 8(%rdi), %r9d")
    cg.emit(f"and ${0xFF & ~((1 << hb) - 1)}, %r9d")
    cg.emit("mov %r8, %r10")
    cg.emit(f"shr ${64 - bo}, %r10")
    cg.emit(f"and ${(1 << hb) - 1}, %r10")
    cg.emit("or %r10, %r9")
    cg.emit("movb %r9b, 8(%rdi)")
    cg.emit("mov %r8, %rax")
    if w < 64:
        cg.emit(f"shl ${64 - w}, %rax")
        cg.emit(f"{'shr' if m.ty.is_unsigned else 'sar'} ${64 - w}, %rax")
