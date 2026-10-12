"""Scalar members of scalar_storage_order("big-endian") structs.

A scalar type flagged `is_rev_order` (see parser/structs.py mark_reverse_scalars)
keeps its value as a big-endian image in memory. Loads byte-reverse the image
into the accumulator (%rax, or %xmm0 for float/double); stores reverse the
value into a scratch register (%rdx) and write it, leaving the value in place
so an assignment still yields its value.
"""
from __future__ import annotations

from hcc.typesys import TypeKind


def load_rev_scalar(cg, ty) -> None:
    """%rax holds the member address -> value in %rax / %xmm0."""
    k, n = ty.kind, ty.size
    if k in (TypeKind.FLOAT, TypeKind.DOUBLE):
        if n == 4:
            cg.emit("movl (%rax), %eax")
            cg.emit("bswap %eax")
            cg.emit("movd %eax, %xmm0")
        else:
            cg.emit("mov (%rax), %rax")
            cg.emit("bswap %rax")
            cg.emit("movq %rax, %xmm0")
        return
    unsigned = ty.is_unsigned or k in (TypeKind.PTR, TypeKind.BOOL)
    if n == 1:
        cg.emit(f"{'movzbl' if unsigned else 'movsbl'} (%rax), %eax")
    elif n == 2:
        cg.emit("movzwl (%rax), %eax")
        cg.emit("rol $8, %ax")
        cg.emit(f"{'movzwl' if unsigned else 'movswl'} %ax, %eax")
    elif n == 4:
        cg.emit("movl (%rax), %eax")
        cg.emit("bswap %eax")
        cg.emit("movslq %eax, %rax")
    else:
        cg.emit("mov (%rax), %rax")
        cg.emit("bswap %rax")


def store_rev_scalar(cg, ty) -> None:
    """Address in %rdi, value in %rax (integer) or %xmm0 (float/double)."""
    k, n = ty.kind, ty.size
    if k in (TypeKind.FLOAT, TypeKind.DOUBLE):
        if n == 4:
            cg.emit("movd %xmm0, %edx")
            cg.emit("bswap %edx")
            cg.emit("movl %edx, (%rdi)")
        else:
            cg.emit("movq %xmm0, %rdx")
            cg.emit("bswap %rdx")
            cg.emit("mov %rdx, (%rdi)")
        return
    if n == 1:
        cg.emit("mov %al, (%rdi)")
    elif n == 2:
        cg.emit("mov %ax, %dx")
        cg.emit("rol $8, %dx")
        cg.emit("mov %dx, (%rdi)")
    elif n == 4:
        cg.emit("mov %eax, %edx")
        cg.emit("bswap %edx")
        cg.emit("mov %edx, (%rdi)")
    else:
        cg.emit("mov %rax, %rdx")
        cg.emit("bswap %rdx")
        cg.emit("mov %rdx, (%rdi)")
