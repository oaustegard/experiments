"""GCC vectors whose elements are 128-bit integers (`unsigned __int128`).

Each element is handled as two 64-bit halves (low at +0, high at +8).
Add/sub use carry/borrow across the halves; bitwise ops act on each half;
shifts use shld/shrd plus a cmov for counts >= 64 (count taken mod 128).
Multiplication is the low 128 bits (mulq + two imul cross terms). Division and
remainder call the libgcc helpers (__udivti3, __divti3, __umodti3, __modti3),
which the gcc driver links in anyway.
"""
from __future__ import annotations

from hcc import diag

_WIDE_OPS = ("+", "-", "*", "&", "|", "^", "<<", ">>", "/", "%")

_DIVMOD_HELPER = {
    ("/", False): "__udivti3",
    ("/", True): "__divti3",
    ("%", False): "__umodti3",
    ("%", True): "__modti3",
}


def _wide_divmod(self, node, size: int, signed: bool) -> None:
    """Per-element 128-bit `/` or `%` via libgcc helpers.

    Entry: %rsi = lhs base, %r8 = rhs base, %rdi = result base.
    The bases are spilled to the stack so they survive the calls. The
    helpers take (rdi:rsi = dividend, rdx:rcx = divisor) and return the
    quotient or remainder in rax:rdx. Padding keeps %rsp 16-byte aligned
    at each call, given the stack machine's depth parity convention."""
    helper = _DIVMOD_HELPER[(node.op, signed)]
    pad = 8 if self.depth % 2 == 0 else 0
    if pad:
        self.emit("sub $8, %rsp")
    self.emit("push %rdi")                       # 16(%rsp) after pushes below
    self.emit("push %r8")                        # 8(%rsp)
    self.emit("push %rsi")                       # 0(%rsp)
    for i in range(0, size, 16):
        self.emit("mov 0(%rsp), %rax")           # lhs base
        self.emit("mov 8(%rsp), %rcx")           # rhs base
        self.emit(f"movq {i}(%rax), %rdi")       # a_lo
        self.emit(f"movq {i + 8}(%rax), %rsi")   # a_hi
        self.emit(f"movq {i}(%rcx), %rdx")       # b_lo
        self.emit(f"movq {i + 8}(%rcx), %r8")    # b_hi
        self.emit("mov %r8, %rcx")
        self.emit(f"call {helper}")
        self.emit("mov 16(%rsp), %rcx")          # result base
        self.emit(f"movq %rax, {i}(%rcx)")
        self.emit(f"movq %rdx, {i + 8}(%rcx)")
    self.emit("mov 16(%rsp), %rax")
    self.emit(f"add ${24 + pad}, %rsp")


def _wide_shift(self, op: str, signed: bool) -> None:
    """Shift %rax:%rdx (lhs) by the low 64 bits of %rcx (rhs), count taken mod 128.
    Result in %rax:%rdx. Clobbers %r10, %r11, %r9 and %cl."""
    self.emit("andq $127, %rcx")
    if op == "<<":
        # n < 64: shld/shl; n >= 64: hi = lo << (n-64), lo = 0
        self.emit("shldq %cl, %rax, %rdx")
        self.emit("shlq %cl, %rax")
        self.emit("xorl %r9d, %r9d")
        self.emit("testq $64, %rcx")
        self.emit("cmovnzq %rax, %rdx")
        self.emit("cmovnzq %r9, %rax")
    else:
        # n < 64: shrd + shift hi; n >= 64: lo = hi >> (n-64), hi = fill
        shift = "sarq" if signed else "shrq"
        self.emit("movq %rdx, %r10")
        self.emit("movq %rdx, %r11")
        self.emit("shrdq %cl, %rdx, %rax")
        self.emit(f"{shift} %cl, %rdx")
        self.emit(f"{shift} %cl, %r10")
        if signed:
            self.emit("sarq $63, %r11")
        else:
            self.emit("xorl %r11d, %r11d")
        self.emit("testq $64, %rcx")
        self.emit("cmovnzq %r10, %rax")
        self.emit("cmovnzq %r11, %rdx")


def _wide_mul(self, i: int) -> None:
    """Low 128 bits of (%rsi+i) * (%r8+i) stored at (%rdi+i).
    The low 128 bits of a product do not depend on signedness.
    lo*lo gives the full 128-bit product; the two cross products only
    contribute their low 64 bits to the high half. Clobbers %rax, %rdx,
    %rcx, %r9, %r10, %r11."""
    self.emit(f"movq {i}(%rsi), %r10")       # a_lo
    self.emit(f"movq {i + 8}(%rsi), %r11")   # a_hi
    self.emit(f"movq {i}(%r8), %rcx")        # b_lo
    self.emit(f"movq {i + 8}(%r8), %r9")     # b_hi
    self.emit("movq %r10, %rax")
    self.emit("mulq %rcx")                   # rdx:rax = a_lo * b_lo
    self.emit("imulq %r9, %r10")             # a_lo * b_hi (low 64)
    self.emit("imulq %rcx, %r11")            # a_hi * b_lo (low 64)
    self.emit("addq %r10, %rdx")
    self.emit("addq %r11, %rdx")
    self.emit(f"movq %rax, {i}(%rdi)")
    self.emit(f"movq %rdx, {i + 8}(%rdi)")


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
        self.frame_lea(node.result_buf.offset, "%rdi")
        signed = not node.lhs.ty.base.is_unsigned
        if node.op in ("/", "%"):
            return _wide_divmod(self, node, size, signed)
        for i in range(0, size, 16):
            if node.op == "*":
                _wide_mul(self, i)
                continue
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
            elif node.op in ("<<", ">>"):
                _wide_shift(self, node.op, signed)
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
        self.frame_lea(node.result_buf.offset, "%rdi")
        for i in range(0, size, 16):
            self.emit(f"movq {i}(%rsi), %rax")
            self.emit(f"movq {i + 8}(%rsi), %rdx")
            self.emit("negq %rax")
            self.emit("adcq $0, %rdx")
            self.emit("negq %rdx")
            self.emit(f"movq %rax, {i}(%rdi)")
            self.emit(f"movq %rdx, {i + 8}(%rdi)")
        self.emit("mov %rdi, %rax")
