"""GCC vectors of _Decimal32/64/128 (`typedef _Decimal64 V __attribute__((vector_size(16)))`).

Each lane is a decimal value, so `a op b` is one libgcc call per lane (see
codegen/decimal.py) and `-v` flips each lane's sign bit. Vector values are
addresses (as for all vectors); results go to the expression's result buffer.
"""
from __future__ import annotations

from hcc.codegen.decimal import DEC_LOAD, DEC_STORE


class DecimalVectorMixin:

    def vector_decimal_binary(self, node) -> None:
        lty = node.lhs.ty
        es = lty.base.size
        self.gen_expr(node.rhs)                  # %rax = rhs address
        self.push()
        self.gen_expr(node.lhs)                  # %rax = lhs address
        self.push()                              # 0(%rsp) lhs, 8(%rsp) rhs
        for i in range(0, lty.size, es):
            self.emit(f"mov 8(%rsp), %rax")
            self.emit(f"{DEC_LOAD[lty.base.kind]} {i}(%rax), %xmm1")
            self.emit(f"mov 0(%rsp), %rax")
            self.emit(f"{DEC_LOAD[lty.base.kind]} {i}(%rax), %xmm0")
            self.decimal_lane_op(node.op, lty.base)
            self.frame_lea(node.result_buf.offset + i, "%rcx")
            self.emit(f"{DEC_STORE[lty.base.kind]} %xmm0, (%rcx)")
        self.emit("add $16, %rsp")
        self.depth -= 2
        self.frame_lea(node.result_buf.offset, "%rax")

    def vector_decimal_neg(self, node) -> None:
        ty = node.operand.ty
        es = ty.base.size
        self.gen_expr(node.operand)              # %rax = source address
        for i in range(0, ty.size, es):
            self.emit(f"{DEC_LOAD[ty.base.kind]} {i}(%rax), %xmm0")
            self.frame_lea(node.result_buf.offset + i, "%rcx")
            self.emit(f"{DEC_STORE[ty.base.kind]} %xmm0, (%rcx)")
            self.emit(f"xorb $0x80, {es - 1}(%rcx)")   # flip the sign bit of the lane
        self.frame_lea(node.result_buf.offset, "%rax")
