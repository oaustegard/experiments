"""Bitfields in scalar_storage_order("big-endian") structs.

A reverse-order bitfield occupies `m.rev_bytes` memory bytes starting at
`m.offset`; those bytes are read/written as one big-endian integer and
`m.bit_offset` is the field's LSB position inside that integer (see
parser/structs.py mirror_reverse_bitfields). Register use: %rsi holds the
span address, %rdx is scratch, %rax the value.
"""
from __future__ import annotations


def emit_load_be_rsi(cg, n: int) -> None:
    """%rax = big-endian integer of the n bytes at (%rsi)."""
    cg.emit("xor %eax, %eax")
    for k in range(n):
        cg.emit(f"movzbl {k}(%rsi), %edx")
        cg.emit("shl $8, %rax")
        cg.emit("or %rdx, %rax")


def emit_store_be_rsi(cg, n: int) -> None:
    """Store the low n bytes of %rax big-endian at (%rsi)."""
    for k in range(n):
        cg.emit("mov %rax, %rdx")
        if n - 1 - k:
            cg.emit(f"shr ${8 * (n - 1 - k)}, %rdx")
        cg.emit(f"mov %dl, {k}(%rsi)")


def load_rev_bitfield(cg, m) -> None:
    """Address of the member in %rax -> span value in %rax."""
    cg.emit("mov %rax, %rsi")
    emit_load_be_rsi(cg, m.rev_bytes)


def assign_rev_bitfield(cg, node) -> None:
    """`lhs = rhs` for a reverse-order bitfield. Called after the lhs address
    was pushed and the rhs value is in %rax; leaves the value in %rax."""
    m = node.lhs.member
    w, bo, n = m.bit_width, m.bit_offset, m.rev_bytes
    mask = (1 << w) - 1
    cg.emit("mov %rax, %r8")                        # value to return
    cg.emit("mov %rax, %rdi")
    cg.emit(f"movabs ${mask}, %rcx")
    cg.emit("and %rcx, %rdi")
    cg.emit(f"shl ${bo}, %rdi")
    cg.pop("%rsi")                                  # span address
    emit_load_be_rsi(cg, n)
    cg.emit(f"movabs ${(~(mask << bo)) & ((1 << 64) - 1)}, %r9")
    cg.emit("and %r9, %rax")
    cg.emit("or %rdi, %rax")
    emit_store_be_rsi(cg, n)
    cg.emit("mov %r8, %rax")
    cg.emit(f"shl ${64 - w}, %rax")
    cg.emit(f"{'shr' if m.ty.is_unsigned else 'sar'} ${64 - w}, %rax")
