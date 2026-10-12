"""Element-wise integer GCC vector arithmetic (CONTRACT section 5).

A vector value is represented by its address (like a struct). Sema and the
parser attach an anonymous `result_buf` local to each vector-valued Binary
(`+ - * / % & | ^`) and Unary (`-`) node. Codegen loads each element of the
operands, widens it to 64 bits, combines them, stores the low bytes into the
result buffer, and leaves %rax = address of the result.
"""
from __future__ import annotations

from hcc.codegen.vector_cmp import vector_compare
from hcc.typesys import is_flonum

_CMP_OPS = ("==", "!=", "<", "<=", ">", ">=")

_FBIN = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

_LOAD_MN = {(1, True): "movsbq", (1, False): "movzbq", (2, True): "movswq",
            (2, False): "movzwq", (4, True): "movslq", (8, True): "movq",
            (8, False): "movq"}
_STORE = {1: ("movb", "%al"), 2: ("movw", "%ax"), 4: ("movl", "%eax"), 8: ("movq", "%rax")}
_BIN = {"+": "add", "-": "sub", "*": "imul", "&": "and", "|": "or", "^": "xor"}


class VectorOpsMixin:

    def _vec_load(self, size: int, signed: bool, off: int, base: str,
                  reg64: str, reg32: str) -> None:
        """Load one element at off(base), widened to 64 bits in reg64."""
        if size == 4 and not signed:
            self.emit(f"movl {off}({base}), {reg32}")   # zero-extends to 64 bits
        else:
            self.emit(f"{_LOAD_MN[(size, signed)]} {off}({base}), {reg64}")

    def vector_float_binary(self, node) -> None:
        """`a op b` on GCC float/double vectors, element by element via SSE."""
        lty = node.lhs.ty
        size, es = lty.size, lty.base.size
        sfx = "ss" if es == 4 else "sd"
        mn = _FBIN[node.op] + sfx
        self.gen_expr(node.rhs)
        self.push()
        self.gen_expr(node.lhs)
        self.emit("mov %rax, %rsi")
        self.pop("%r8")
        self.frame_lea(node.result_buf.offset, "%rdi")
        mov = "movss" if es == 4 else "movsd"
        for i in range(0, size, es):
            self.emit(f"{mov} {i}(%rsi), %xmm0")
            self.emit(f"{mov} {i}(%r8), %xmm1")
            self.emit(f"{mn} %xmm1, %xmm0")
            self.emit(f"{mov} %xmm0, {i}(%rdi)")
        self.emit("mov %rdi, %rax")

    def vector_binary(self, node) -> None:
        lty = node.lhs.ty
        if node.op in _CMP_OPS:
            return vector_compare(self, node)
        if is_flonum(lty.base):
            return self.vector_float_binary(node)
        if lty.base.size == 16:
            return self.vector_wide_binary(node)
        size, es = lty.size, lty.base.size
        signed = not lty.base.is_unsigned
        self.gen_expr(node.rhs)                      # %rax = rhs address
        self.push()
        self.gen_expr(node.lhs)                      # %rax = lhs address
        self.emit("mov %rax, %rsi")
        self.pop("%r8")
        self.frame_lea(node.result_buf.offset, "%rdi")
        for i in range(0, size, es):
            self._vec_load(es, signed, i, "%rsi", "%rax", "%eax")
            self._vec_load(es, signed, i, "%r8", "%rcx", "%ecx")
            op = node.op
            if op in _BIN:
                self.emit(f"{_BIN[op]} %rcx, %rax")
            elif op in ("/", "%"):
                if signed:
                    self.emit("cqo")
                    self.emit("idiv %rcx")
                else:
                    self.emit("xor %edx, %edx")
                    self.emit("div %rcx")
                if op == "%":
                    self.emit("mov %rdx, %rax")
            elif op == "<<":
                self.emit("shl %cl, %rax")
            elif op == ">>":
                self.emit("sar %cl, %rax" if signed else "shr %cl, %rax")
            mn, reg = _STORE[es]
            self.emit(f"{mn} {reg}, {i}(%rdi)")
        self.emit("mov %rdi, %rax")

    def vector_neg(self, node) -> None:
        """`-v` on a GCC integer vector: negate each element into the result buffer."""
        ty = node.operand.ty
        if ty.base.size == 16:
            return self.vector_wide_neg(node)
        size, es = ty.size, ty.base.size
        signed = not ty.base.is_unsigned
        self.gen_expr(node.operand)                  # %rax = source address
        self.emit("mov %rax, %rsi")
        self.frame_lea(node.result_buf.offset, "%rdi")
        for i in range(0, size, es):
            self._vec_load(es, signed, i, "%rsi", "%rax", "%eax")
            self.emit("neg %rax")
            mn, reg = _STORE[es]
            self.emit(f"{mn} {reg}, {i}(%rdi)")
        self.emit("mov %rdi, %rax")
