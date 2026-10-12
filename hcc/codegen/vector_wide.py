"""GCC vectors whose elements are 128-bit integers (`unsigned __int128`).

Each element is handled as two 64-bit halves (low at +0, high at +8).
Add/sub use carry/borrow across the halves; bitwise ops act on each half.
Multiplication, division, remainder and shifts need a real 128-bit
multiplier/divider and are reported as unsupported rather than miscompiled.
"""
from __future__ import annotations

from hcc import diag

_WIDE_OPS = ("+", "-", "&", "|", "^")


class VectorWideMixin:

    def vector_wide_binary(self, node) -> None:
        """`a op b` on vectors of 16-byte integer elements."""
        if node.op not in _WIDE_OPS:
            diag.error(node, f"codegen: vector operator '{node.op}' on "
                       "128-bit elements is not supported", "codegen")
        size = node.lhs.ty.size
        self.gen_expr(node.rhs)                      # %rax = rhs address
        self.push()
        self.gen_expr(node.lhs)                      # %rax = lhs address
        self.emit("mov %rax, %rsi")
        self.pop("%r8")
        self.emit(f"lea {node.result_buf.offset}(%rbp), %rdi")
        for i in range(0, size, 16):
            self.emit(f"movq {i}(%rsi), %rax")
            self.emit(f"movq {i + 8}(%rsi), %rdx")
            self.emit(f"movq {i}(%r8), %rcx")
            self.emit(f"movq {i + 8}(%r8), %r9")
            if node.op == "+":
                self.emit("addq %rcx, %rax")
                self.emit("adcq %r9, %rdx")
            elif node.op == "-":
                self.emit("subq %rcx, %rax")
                self.emit("sbbq %r9, %rdx")
            else:
                mn = {"&": "and", "|": "or", "^": "xor"}[node.op]
                self.emit(f"{mn}q %rcx, %rax")
                self.emit(f"{mn}q %r9, %rdx")
            self.emit(f"movq %rax, {i}(%rdi)")
            self.emit(f"movq %rdx, {i + 8}(%rdi)")
        self.emit("mov %rdi, %rax")

    def vector_wide_neg(self, node) -> None:
        """`-v` on a vector of 128-bit elements (two's complement per element)."""
        size = node.operand.ty.size
        self.gen_expr(node.operand)                  # %rax = source address
        self.emit("mov %rax, %rsi")
        self.emit(f"lea {node.result_buf.offset}(%rbp), %rdi")
        for i in range(0, size, 16):
            self.emit(f"movq {i}(%rsi), %rax")
            self.emit(f"movq {i + 8}(%rsi), %rdx")
            self.emit("negq %rax")
            self.emit("adcq $0, %rdx")
            self.emit("negq %rdx")
            self.emit(f"movq %rax, {i}(%rdi)")
            self.emit(f"movq %rdx, {i + 8}(%rdi)")
        self.emit("mov %rdi, %rax")
