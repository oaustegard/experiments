"""_Complex long double (x87 parts, 16 bytes each) codegen.

Parts live in memory as 80-bit values at offsets 0 and 16. Arithmetic on
+ - == != is inline x87; * and / call libgcc's __multc3 / __divtc3 with the
four long doubles on the stack (SysV memory class). A complex long double
result is returned in %st(0) (real) and %st(1) (imaginary).
"""
from __future__ import annotations

from hcc.sema.initializers import float_to_x87
from hcc.typesys import TypeKind


def ld_lit(gen, node) -> None:
    buf = node.result_buf.offset
    for off, v in ((0, node.re), (16, node.im)):
        raw = float_to_x87(v)
        lo = int.from_bytes(raw[:8], "little")
        hi = int.from_bytes(raw[8:10], "little")
        gen.emit(f"movabs ${lo}, %rcx")
        gen.emit(f"mov %rcx, {buf + off}(%rbp)")
        gen.emit(f"movw ${hi}, {buf + off + 8}(%rbp)")
    gen.emit(f"lea {buf}(%rbp), %rax")


def ld_binary(gen, node) -> None:
    op = node.op
    gen.gen_expr(node.rhs)
    gen.push()
    gen.gen_expr(node.lhs)
    gen.pop("%rdi")                          # %rax = lhs address, %rdi = rhs address
    if op in ("==", "!="):
        _equal(gen)
        if op == "!=":
            gen.emit("xor $1, %eax")
        return
    if op in ("+", "-"):
        gen.emit(f"lea {node.result_buf.offset}(%rbp), %rsi")
        for off in (0, 16):                  # a + b, or a + (-b) (exact, like a - b)
            gen.emit(f"fldt {off}(%rax)")
            gen.emit(f"fldt {off}(%rdi)")
            if op == "-":
                gen.emit("fchs")
            gen.emit("faddp")
            gen.emit(f"fstpt {off}(%rsi)")
        gen.emit("mov %rsi, %rax")
        return
    # * and /: libgcc helper with (a, b, c, d) as four 16-byte stack arguments
    helper = "__mulxc3" if op == "*" else "__divxc3"      # libgcc names for x87 (XFmode) parts
    pad = 8 if gen.depth % 2 == 1 else 0     # 64 bytes of arguments keep alignment
    if pad:
        gen.emit("sub $8, %rsp")
    gen.emit("sub $64, %rsp")
    for slot, (base, off) in enumerate((("%rax", 0), ("%rax", 16), ("%rdi", 0), ("%rdi", 16))):
        gen.emit(f"fldt {off}({base})")
        gen.emit(f"fstpt {slot * 16}(%rsp)")
    gen.emit(f"call {helper}")
    gen.emit("add $64, %rsp")
    if pad:
        gen.emit("add $8, %rsp")
    gen.emit(f"lea {node.result_buf.offset}(%rbp), %rax")
    gen.emit("fstpt 0(%rax)")                # st(0) = real part
    gen.emit("fstpt 16(%rax)")               # then imaginary part


def _equal(gen) -> None:
    """%eax = 1 iff both parts compare equal (unordered parts are unequal)."""
    for i, off in enumerate((0, 16)):
        gen.emit(f"fldt {off}(%rax)")
        gen.emit(f"fldt {off}(%rdi)")
        gen.emit("fucomip %st(1), %st")
        gen.emit("fstp %st(0)")
        gen.emit("sete %r8b" if i == 0 else "sete %r10b")
        gen.emit("setnp %r9b")
        gen.emit("and %r9b, %r8b" if i == 0 else "and %r9b, %r10b")
    gen.emit("and %r10b, %r8b")
    gen.emit("movzbl %r8b, %eax")


def ld_neg(gen, node) -> None:
    gen.gen_expr(node.operand)
    gen.emit(f"lea {node.result_buf.offset}(%rbp), %rsi")
    for off in (0, 16):
        gen.emit(f"fldt {off}(%rax)")
        gen.emit("fchs")
        gen.emit(f"fstpt {off}(%rsi)")
    gen.emit("mov %rsi, %rax")


def ld_cmp_zero(gen) -> None:
    """Flags as `cmp $0`: ZF=1 iff both parts are zero (%rax = address)."""
    gen.emit("mov (%rax), %rdx")
    gen.emit("movzwl 8(%rax), %ecx")
    gen.emit("and $0x7fff, %ecx")
    gen.emit("or %rcx, %rdx")
    gen.emit("mov 16(%rax), %r8")
    gen.emit("movzwl 24(%rax), %ecx")
    gen.emit("and $0x7fff, %ecx")
    gen.emit("or %rcx, %r8")
    gen.emit("or %r8, %rdx")
    gen.emit("test %rdx, %rdx")


def ld_return(gen) -> None:
    """Return the complex long double at %rax: real in st(0), imaginary in st(1)."""
    gen.emit("fldt 16(%rax)")
    gen.emit("fldt (%rax)")


def ld_store_result(gen, buf_off: int) -> None:
    """After a call returning complex long double (st(0)=real, st(1)=imag)."""
    gen.emit(f"fstpt {buf_off}(%rbp)")
    gen.emit(f"fstpt {buf_off + 16}(%rbp)")


def is_ld_complex(ty) -> bool:
    return ty.kind == TypeKind.COMPLEX and ty.base.kind == TypeKind.LDOUBLE


def part_load(gen, base, addr: str, off: int) -> None:
    """Load a float/double/long double part: long double to st(0), others to %xmm0."""
    if base.size == 16:
        gen.emit(f"fldt {off}({addr})")
    else:
        gen.emit(f"{'movss' if base.size == 4 else 'movsd'} {off}({addr}), %xmm0")


def part_store(gen, base, off: int) -> None:
    """Store the value produced by part_load's convention into the buffer at %rsi."""
    if base.size == 16:
        gen.emit(f"fstpt {off}(%rsi)")
    else:
        gen.emit(f"{'movss' if base.size == 4 else 'movsd'} %xmm0, {off}(%rsi)")


def ld_cast(gen, node) -> None:
    """Conversions between complex long double and complex float/double (or real)."""
    frm, to = node.operand.ty, node.ty
    gen.gen_expr(node.operand)
    buf = node.result_buf.offset
    if frm.kind != TypeKind.COMPLEX:        # real -> complex: imaginary part +0
        gen.cast(frm, to.base)
        gen.emit(f"lea {buf}(%rbp), %rsi")
        part_store(gen, to.base, 0)
        n = to.base.size                    # zero the imaginary part
        if n == 16:
            gen.emit("movq $0, 16(%rsi)")
            gen.emit("movq $0, 24(%rsi)")
        elif n == 8:
            gen.emit("movq $0, 8(%rsi)")
        else:
            gen.emit("movl $0, 4(%rsi)")
        gen.emit("mov %rsi, %rax")
        return
    gen.emit("mov %rax, %r10")
    for i in (0, 1):
        src, dst = frm.base, to.base
        part_load(gen, src, "%r10", i * src.size)
        if src.kind != dst.kind:
            gen.cast(src, dst)
        gen.emit(f"lea {buf}(%rbp), %rsi")
        part_store(gen, dst, i * dst.size)
    gen.emit("mov %rsi, %rax")
