"""_Complex <integer> (GNU extension): codegen for integer-part complex values.

Same representation as float complex (address in %rax, parts at offset 0 and
sizeof(T)); arithmetic is inline integer code. Division follows GCC's lowering:
(a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c*c+d*d) with truncating integer division.
"""
from __future__ import annotations

from hcc.typesys import TypeKind

_STORE_MOV = {1: "movb", 2: "movw", 4: "movl", 8: "movq"}
_STORE_REG = {1: "%al", 2: "%ax", 4: "%eax", 8: "%rax"}


def is_int_part(ty) -> bool:
    """True for _Complex <integer>; `ty` is the complex type."""
    return ty.base.kind not in (TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LDOUBLE)


def _load_at(gen, base, addr: str, off: int) -> None:
    """%rax = integer part at `off(addr)`, extended to 64 bits (value semantics)."""
    gen.emit(f"lea {off}({addr}), %rax")
    gen.load(base)
    if base.size == 4 and base.is_unsigned:
        gen.emit("mov %eax, %eax")          # load() sign-extends 32-bit values


def _store_rax(gen, base, off: int) -> None:
    gen.emit(f"mov {_STORE_REG[base.size]}, {off}(%rsi)")


def _zero_part(gen, base, off: int) -> None:
    gen.emit(f"{_STORE_MOV[base.size]} $0, {off}(%rsi)")


def int_binary(gen, node) -> None:
    """Binary + - * / == != on _Complex <integer> operands (result in %rax)."""
    op = node.op
    base = node.lhs.ty.base
    es = base.size
    gen.gen_expr(node.rhs)
    gen.push()
    gen.gen_expr(node.lhs)
    gen.pop("%rdi")
    gen.emit("mov %rax, %r10")              # lhs address
    gen.emit("mov %rdi, %r11")              # rhs address
    if op in ("==", "!="):
        _equal(gen, base, es, op)
        return
    gen.emit(f"lea {node.result_buf.offset}(%rbp), %rsi")
    if op in ("+", "-"):
        for off in (0, es):
            _load_at(gen, base, "%r11", off)
            gen.emit("mov %rax, %rdx")
            _load_at(gen, base, "%r10", off)
            gen.emit("add %rdx, %rax" if op == "+" else "sub %rdx, %rax")
            _store_rax(gen, base, off)
        gen.emit("mov %rsi, %rax")
        return
    # a+bi in r8,r9 ; c+di in rcx,rdi
    _load_at(gen, base, "%r10", 0)
    gen.emit("mov %rax, %r8")
    _load_at(gen, base, "%r10", es)
    gen.emit("mov %rax, %r9")
    _load_at(gen, base, "%r11", 0)
    gen.emit("mov %rax, %rcx")
    _load_at(gen, base, "%r11", es)
    gen.emit("mov %rax, %rdi")
    if op == "*":
        gen.emit("mov %r8, %rax")           # re = ac - bd
        gen.emit("imul %rcx, %rax")
        gen.emit("mov %r9, %rdx")
        gen.emit("imul %rdi, %rdx")
        gen.emit("sub %rdx, %rax")
        _store_rax(gen, base, 0)
        gen.emit("mov %r8, %rax")           # im = ad + bc
        gen.emit("imul %rdi, %rax")
        gen.emit("mov %r9, %rdx")
        gen.emit("imul %rcx, %rdx")
        gen.emit("add %rdx, %rax")
        _store_rax(gen, base, es)
        gen.emit("mov %rsi, %rax")
        return
    # division: den (r10) = c*c + d*d ; re num (r11) = ac + bd ; im num (rcx) = bc - ad
    gen.emit("mov %rcx, %rax")
    gen.emit("imul %rcx, %rax")
    gen.emit("mov %rdi, %rdx")
    gen.emit("imul %rdi, %rdx")
    gen.emit("add %rdx, %rax")
    gen.emit("mov %rax, %r10")
    gen.emit("mov %r8, %rax")
    gen.emit("imul %rcx, %rax")
    gen.emit("mov %r9, %rdx")
    gen.emit("imul %rdi, %rdx")
    gen.emit("add %rdx, %rax")
    gen.emit("mov %rax, %r11")
    gen.emit("mov %r9, %rax")
    gen.emit("imul %rcx, %rax")
    gen.emit("mov %r8, %rdx")
    gen.emit("imul %rdi, %rdx")
    gen.emit("sub %rdx, %rax")
    gen.emit("mov %rax, %rcx")
    _divide(gen, base, "%r11", 0)
    _divide(gen, base, "%rcx", es)
    gen.emit("mov %rsi, %rax")


def _divide(gen, base, num: str, off: int) -> None:
    """Store num / den (den in %r10) to the result part at `off`."""
    gen.emit(f"mov {num}, %rax")
    if base.is_unsigned:
        gen.emit("xor %edx, %edx")
        gen.emit("div %r10")
    else:
        gen.emit("cqo")
        gen.emit("idiv %r10")
    _store_rax(gen, base, off)


def _equal(gen, base, es, op) -> None:
    _load_at(gen, base, "%r10", 0)
    gen.emit("mov %rax, %rdx")
    _load_at(gen, base, "%r11", 0)
    gen.emit("cmp %rax, %rdx")
    gen.emit("sete %r8b")
    _load_at(gen, base, "%r10", es)
    gen.emit("mov %rax, %rdx")
    _load_at(gen, base, "%r11", es)
    gen.emit("cmp %rax, %rdx")
    gen.emit("sete %r9b")
    gen.emit("and %r9b, %r8b")
    gen.emit("movzbl %r8b, %eax")
    if op == "!=":
        gen.emit("xor $1, %eax")


def int_neg(gen, node) -> None:
    base = node.ty.base
    es = base.size
    gen.gen_expr(node.operand)
    gen.emit("mov %rax, %r10")
    gen.emit(f"lea {node.result_buf.offset}(%rbp), %rsi")
    for off in (0, es):
        _load_at(gen, base, "%r10", off)
        gen.emit("neg %rax")
        _store_rax(gen, base, off)
    gen.emit("mov %rsi, %rax")


def int_cmp_zero(gen, ty) -> None:
    """Flags as `cmp $0`: ZF=1 iff both integer parts are zero (%rax = address)."""
    base = ty.base
    gen.emit("mov %rax, %r10")
    _load_at(gen, base, "%r10", 0)
    gen.emit("mov %rax, %rdx")
    _load_at(gen, base, "%r10", base.size)
    gen.emit("or %rdx, %rax")


def int_cast(gen, node) -> None:
    """Conversions where a _Complex <integer> is source and/or target."""
    frm, to = node.operand.ty, node.ty
    if frm.kind != TypeKind.COMPLEX:          # real -> _Complex <integer>
        gen.gen_expr(node.operand)
        gen.cast(frm, to.base)                # value in %rax (int) or %xmm0 (float)
        gen.emit(f"lea {node.result_buf.offset}(%rbp), %rsi")
        if to.base.kind in (TypeKind.FLOAT, TypeKind.DOUBLE):
            gen.emit(f"movsd %xmm0, 0(%rsi)" if to.base.size == 8 else "movss %xmm0, 0(%rsi)")
            gen.emit(f"{_STORE_MOV[to.base.size]} $0, {to.base.size}(%rsi)")
        else:
            _store_rax(gen, to.base, 0)
            _zero_part(gen, to.base, to.base.size)
        gen.emit("mov %rsi, %rax")
        return
    gen.gen_expr(node.operand)                # complex -> real or complex
    if to.kind != TypeKind.COMPLEX:
        gen.load(frm.base)
        gen.cast(frm.base, to)
        return
    gen.emit("mov %rax, %r10")
    gen.emit(f"lea {node.result_buf.offset}(%rbp), %rsi")
    for i in (0, 1):
        src, dst = frm.base, to.base
        foff, toff = i * src.size, i * dst.size
        _load_part(gen, src, "%r10", foff)
        if src.kind != dst.kind or src.size != dst.size:
            gen.cast(src, dst)
        if dst.kind in (TypeKind.FLOAT, TypeKind.DOUBLE):
            gen.emit(f"{'movss' if dst.size == 4 else 'movsd'} %xmm0, {toff}(%rsi)")
        else:
            _store_rax(gen, dst, toff)
    gen.emit("mov %rsi, %rax")


def _load_part(gen, base, addr: str, off: int) -> None:
    """Load a part of any scalar kind: integers to %rax, floats to %xmm0."""
    if base.kind in (TypeKind.FLOAT, TypeKind.DOUBLE):
        gen.emit(f"{'movss' if base.size == 4 else 'movsd'} {off}({addr}), %xmm0")
    else:
        _load_at(gen, base, addr, off)
