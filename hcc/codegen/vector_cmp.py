"""Element-wise integer GCC vector comparisons (CONTRACT section 5).

`a OP b` on two integer vectors yields a vector whose lanes are -1 (true)
or 0 (false). Each lane is compared with the signedness of the element type.
16-byte elements (__int128) are compared as two 64-bit halves.
"""
from __future__ import annotations

_EQ_CC = {"==": "e", "!=": "ne"}
# signed / unsigned condition codes for a - b (AT&T `cmp b, a`)
_ORD_CC = {"<": ("l", "b"), "<=": ("le", "be"), ">": ("g", "a"), ">=": ("ge", "ae")}
_STORE = {1: "movb %al, {off}(%rdi)", 2: "movw %ax, {off}(%rdi)",
          4: "movl %eax, {off}(%rdi)", 8: "movq %rax, {off}(%rdi)"}
_LOAD_MN = {(1, True): "movsbq", (1, False): "movzbq", (2, True): "movswq",
            (2, False): "movzwq", (4, True): "movslq", (8, True): "movq",
            (8, False): "movq"}


def _load(gen, size: int, signed: bool, off: int, base: str, reg64: str, reg32: str) -> None:
    if size == 4 and not signed:
        gen.emit(f"movl {off}({base}), {reg32}")
    else:
        gen.emit(f"{_LOAD_MN[(size, signed)]} {off}({base}), {reg64}")


def _cmp_wide(gen, op: str, off: int, signed: bool) -> None:
    """16-byte lane: leaves the 0/1 result in %al."""
    gen.emit(f"movq {off}(%rsi), %rax")
    gen.emit(f"movq {off + 8}(%rsi), %rdx")
    gen.emit(f"movq {off}(%r8), %rcx")
    gen.emit(f"movq {off + 8}(%r8), %r9")
    if op in _EQ_CC:
        cc = _EQ_CC[op]
        gen.emit("cmp %rcx, %rax")
        gen.emit(f"set{cc} %al")
        gen.emit("cmp %r9, %rdx")
        gen.emit(f"set{cc} %dl")
        gen.emit("and %dl, %al" if op == "==" else "or %dl, %al")
        return
    # 128-bit ordering: subtract with borrow (minuend M, subtrahend S).
    # a > b and a <= b are evaluated as b < a and b >= a.
    if op in (">", "<="):
        m_lo, m_hi, s_lo, s_hi = "%rcx", "%r9", "%rax", "%rdx"
    else:
        m_lo, m_hi, s_lo, s_hi = "%rax", "%rdx", "%rcx", "%r9"
    gen.emit(f"cmp {s_lo}, {m_lo}")
    gen.emit(f"sbb {s_hi}, {m_hi}")
    signed_cc, unsigned_cc = ("l", "b") if op in ("<", ">") else ("ge", "ae")
    gen.emit(f"set{signed_cc if signed else unsigned_cc} %al")


def vector_compare(gen, node) -> None:
    """Lower a vector comparison node; result buffer holds -1/0 lanes."""
    lty = node.lhs.ty
    size, es = lty.size, lty.base.size
    signed = not lty.base.is_unsigned
    op = node.op
    gen.gen_expr(node.rhs)                       # %rax = rhs address
    gen.push()
    gen.gen_expr(node.lhs)                       # %rax = lhs address
    gen.emit("mov %rax, %rsi")
    gen.pop("%r8")
    gen.emit(f"lea {node.result_buf.offset}(%rbp), %rdi")
    for i in range(0, size, es):
        if es == 16:
            _cmp_wide(gen, op, i, signed)
        else:
            _load(gen, es, signed, i, "%rsi", "%rax", "%eax")
            _load(gen, es, signed, i, "%r8", "%rcx", "%ecx")
            gen.emit("cmp %rcx, %rax")
            if op in _EQ_CC:
                gen.emit(f"set{_EQ_CC[op]} %al")
            else:
                gen.emit(f"set{_ORD_CC[op][0 if signed else 1]} %al")
        gen.emit("movzbq %al, %rax")
        gen.emit("neg %rax")                     # 1 -> -1 (all ones), 0 -> 0
        if es == 16:
            gen.emit(f"movq %rax, {i}(%rdi)")
            gen.emit(f"movq %rax, {i + 8}(%rdi)")
        else:
            gen.emit(_STORE[es].format(off=i))
    gen.emit("mov %rdi, %rax")
