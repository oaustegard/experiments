"""Assignment between complex values whose parts have different storage order.

`_Complex T` is two scalars. When one side is a member of a
scalar_storage_order("big-endian") struct its parts are byte-reversed in
memory (Type.is_rev_order on the part type), while a plain complex value
(e.g. a constant expression held in a temporary) is native. A whole-struct
copy would move the bytes unchanged, so the parts are converted one by one:
each part is byte-reversed while copied.
"""
from __future__ import annotations

from hcc import diag


def needs_part_swap(lhs_ty, rhs_ty) -> bool:
    return (lhs_ty.is_complex and rhs_ty.is_complex
            and lhs_ty.base.is_rev_order != rhs_ty.base.is_rev_order)


def assign_complex_rev(cg, node) -> None:
    """Assign node.rhs (complex, other storage order) to node.lhs; leaves %rax
    = source address, like copy_struct does."""
    part = node.lhs.ty.base
    n = part.size
    if n not in (4, 8):
        diag.error(node, "scalar_storage_order: unsupported complex member type", "codegen")
    cg.gen_addr(node.lhs)
    cg.push()
    cg.gen_expr(node.rhs)
    cg.pop("%rdi")
    for off in (0, n):
        if n == 4:
            cg.emit(f"movl {off}(%rax), %edx")
            cg.emit("bswap %edx")
            cg.emit(f"movl %edx, {off}(%rdi)")
        else:
            cg.emit(f"movq {off}(%rax), %rdx")
            cg.emit("bswap %rdx")
            cg.emit(f"movq %rdx, {off}(%rdi)")
