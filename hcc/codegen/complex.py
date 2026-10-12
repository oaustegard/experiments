"""_Complex values (C99 6.2.5, Annex G): codegen.

A complex rvalue is the ADDRESS of its two parts in %rax (like structs), held in
the node's `result_buf` (see parser/complex_temps.py). Parts are T at offset 0
(real) and sizeof(T) (imaginary), T = float or double (_Complex long double is
rejected by the parser).

Arithmetic: + - are inline; * and / call libgcc's __mulXc3 / __divXc3 (the same
helpers gcc uses) so NaN/infinity recovery matches C Annex G.
Calling convention for complex values is SysV: _Complex double returns in
%xmm0:%xmm1, _Complex float packs both parts in the low 8 bytes of %xmm0.
"""
from __future__ import annotations

import struct

from hcc import diag
from hcc.codegen import complex_int as CI
from hcc.codegen import complex_ld as LD
from hcc.floatbits import pack_float32, pack_float64
from hcc.typesys import TypeKind


def _is_f32(ty) -> bool:
    return ty.base.kind == TypeKind.FLOAT


def _mv(ty) -> tuple[str, str]:
    """(move mnemonic, arithmetic suffix) for a complex type's part type."""
    return ("movss", "ss") if _is_f32(ty) else ("movsd", "sd")


class ComplexMixin:

    def expr_ComplexLit(self, node) -> None:
        if LD.is_ld_complex(node.ty):
            return LD.ld_lit(self, node)
        elem = node.ty.base
        buf = node.result_buf.offset
        for off, v in ((0, node.re), (elem.size, node.im)):
            if _is_f32(node.ty):
                bits = struct.unpack("<I", pack_float32(v))[0]
                bits = bits - (1 << 32) if bits >= 1 << 31 else bits
                self.emit(f"movl ${bits}, {buf + off}(%rbp)")
            else:
                bits = struct.unpack("<Q", pack_float64(v))[0]
                self.emit(f"movabs ${bits}, %rcx")
                self.emit(f"mov %rcx, {buf + off}(%rbp)")
        self.emit(f"lea {buf}(%rbp), %rax")

    def expr_ComplexPart(self, node) -> None:
        self.gen_expr(node.operand)              # %rax = address of the complex value
        if node.imag:
            self.emit(f"add ${node.operand.ty.base.size}, %rax")
        self.load(node.ty)

    def complex_binary(self, node) -> None:
        """Binary op on two complex operands (already converted to one type)."""
        op = node.op
        ty = node.lhs.ty
        if CI.is_int_part(ty):
            return CI.int_binary(self, node)
        if LD.is_ld_complex(ty):
            return LD.ld_binary(self, node)
        mv, sfx = _mv(ty)
        es = ty.base.size
        self.gen_expr(node.rhs)
        self.push()
        self.gen_expr(node.lhs)
        self.pop("%rdi")                    # %rax = lhs address, %rdi = rhs address
        if op in ("==", "!="):
            self._complex_eq(mv, sfx, es)
            if op == "!=":
                self.emit("xor $1, %eax")
            return
        buf = node.result_buf.offset
        if op in ("+", "-"):
            self.emit(f"lea {buf}(%rbp), %rsi")
            opn = "add" if op == "+" else "sub"
            for off in (0, es):
                self.emit(f"{mv} {off}(%rax), %xmm0")
                self.emit(f"{opn}{sfx} {off}(%rdi), %xmm0")
                self.emit(f"{mv} %xmm0, {off}(%rsi)")
            self.emit("mov %rsi, %rax")
            return
        if op not in ("*", "/"):
            from hcc import diag
            diag.error(node, f"invalid operands to binary '{op}' (complex)", "codegen")
        helper = f"__{'mul' if op == '*' else 'div'}{'sc' if _is_f32(ty) else 'dc'}3"
        self.emit(f"{mv} 0(%rax), %xmm0")
        self.emit(f"{mv} {es}(%rax), %xmm1")
        self.emit(f"{mv} 0(%rdi), %xmm2")
        self.emit(f"{mv} {es}(%rdi), %xmm3")
        pad = 8 if self.depth % 2 == 1 else 0       # rsp must be 16-aligned at the call
        if pad:
            self.emit("sub $8, %rsp")
        self.emit(f"call {helper}")
        if pad:
            self.emit("add $8, %rsp")
        self.emit(f"lea {buf}(%rbp), %rsi")
        if _is_f32(ty):
            self.emit("movq %xmm0, (%rsi)")          # both floats are in the low 8 bytes
        else:
            self.emit("movsd %xmm0, (%rsi)")
            self.emit("movsd %xmm1, 8(%rsi)")
        self.emit("mov %rsi, %rax")

    def _complex_eq(self, mv: str, sfx: str, es: int) -> None:
        """%eax = 1 iff both parts compare equal (unordered parts are unequal)."""
        self.emit(f"{mv} 0(%rax), %xmm0")
        self.emit(f"ucomi{sfx} 0(%rdi), %xmm0")
        self.emit("sete %r8b")
        self.emit("setnp %r9b")
        self.emit("and %r9b, %r8b")
        self.emit(f"{mv} {es}(%rax), %xmm0")
        self.emit(f"ucomi{sfx} {es}(%rdi), %xmm0")
        self.emit("sete %r10b")
        self.emit("setnp %r9b")
        self.emit("and %r9b, %r10b")
        self.emit("and %r10b, %r8b")
        self.emit("movzbl %r8b, %eax")

    def complex_neg(self, node) -> None:
        ty = node.ty
        if CI.is_int_part(ty):
            return CI.int_neg(self, node)
        if LD.is_ld_complex(ty):
            return LD.ld_neg(self, node)
        es = ty.base.size
        self.gen_expr(node.operand)          # %rax = operand address
        buf = node.result_buf.offset
        self.emit(f"lea {buf}(%rbp), %rsi")
        for off in (0, es):
            if _is_f32(ty):
                self.emit(f"mov {off}(%rax), %ecx")
                self.emit("xor $0x80000000, %ecx")
                self.emit(f"mov %ecx, {off}(%rsi)")
            else:
                self.emit(f"mov {off}(%rax), %rcx")
                self.emit("btc $63, %rcx")
                self.emit(f"mov %rcx, {off}(%rsi)")
        self.emit("mov %rsi, %rax")

    def complex_cast(self, node) -> None:
        frm, to = node.operand.ty, node.ty
        buf = node.result_buf.offset if node.result_buf is not None else None
        if to.kind == TypeKind.BOOL:             # nonzero if either part is nonzero
            self.gen_expr(node.operand)
            self.complex_cmp_zero(frm)
            self.emit("setne %al")
            self.emit("movzbl %al, %eax")
            return
        if (frm.kind == TypeKind.COMPLEX and CI.is_int_part(frm)) or \
                (to.kind == TypeKind.COMPLEX and CI.is_int_part(to)):
            return CI.int_cast(self, node)
        if (LD.is_ld_complex(frm) or LD.is_ld_complex(to)) and to.kind == TypeKind.COMPLEX:
            return LD.ld_cast(self, node)
        if frm.kind == TypeKind.COMPLEX and to.kind == TypeKind.COMPLEX:
            self.gen_expr(node.operand)
            self.emit("mov %rax, %r10")
            fmv, _ = _mv(frm)
            tmv, _ = _mv(to)
            fes, tes = frm.base.size, to.base.size
            for i in (0, 1):                     # real part, then imaginary part
                self.emit(f"{fmv} {i * fes}(%r10), %xmm0")
                if frm.base.kind != to.base.kind:
                    self.cast(frm.base, to.base)
                self.emit(f"lea {buf}(%rbp), %rsi")
                self.emit(f"{tmv} %xmm0, {i * tes}(%rsi)")
            self.emit("mov %rsi, %rax")
            return
        if frm.kind == TypeKind.COMPLEX:         # complex -> real keeps the real part
            self.gen_expr(node.operand)
            self.load(frm.base)
            self.cast(frm.base, to)
            return
        # real -> complex: imaginary part +0
        self.gen_expr(node.operand)
        self.cast(frm, to.base)
        self.emit(f"lea {buf}(%rbp), %rsi")
        mv, _ = _mv(to)
        self.emit(f"{mv} %xmm0, 0(%rsi)")
        self.emit(f"movl $0, {to.base.size}(%rsi)" if _is_f32(to) else f"movq $0, {to.base.size}(%rsi)")
        self.emit("mov %rsi, %rax")

    def complex_cmp_zero(self, ty) -> None:
        """Set flags as `cmp $0` would: ZF=1 iff both parts are zero (%rax = address)."""
        if CI.is_int_part(ty):
            return CI.int_cmp_zero(self, ty)
        if LD.is_ld_complex(ty):
            return LD.ld_cmp_zero(self)
        if _is_f32(ty):
            self.emit("movabs $0x7fffffff7fffffff, %rcx")
            self.emit("mov (%rax), %rdx")
            self.emit("and %rcx, %rdx")
        else:
            self.emit("movabs $0x7fffffffffffffff, %rcx")
            self.emit("mov (%rax), %rdx")
            self.emit("and %rcx, %rdx")
            self.emit("mov 8(%rax), %r8")
            self.emit("and %rcx, %r8")
            self.emit("or %r8, %rdx")
        self.emit("test %rdx, %rdx")
