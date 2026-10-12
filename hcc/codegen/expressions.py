"""Expression codegen: literals, variables, unary ops, assignment, ?:, comma,
logical ops, casts, member access (incl. bitfields), statement expressions.

ExpressionMixin: one `expr_<NodeClass>` method per node. Binary arithmetic is
in arith.py (integers) and floats.py; calls in calls.py; builtins in builtins.py.
"""
from __future__ import annotations

import struct

from hcc.sema.initializers import float_to_x87
from hcc.typesys import TypeKind, is_flonum


class ExpressionMixin:

    def expr_IntLit(self, node) -> None:
        if is_flonum(node.ty):
            return self.expr_FloatLit(node)
        v = node.value
        if -(1 << 31) <= v < (1 << 31):
            self.emit(f"mov ${v}, %rax")
        else:
            self.emit(f"movabs ${v & ((1 << 64) - 1)}, %rax")

    def expr_FloatLit(self, node) -> None:
        k = node.ty.kind
        v = float(node.value)
        if k == TypeKind.FLOAT:
            bits = struct.unpack("<I", struct.pack("<f", v))[0]
            self.emit(f"mov ${bits}, %eax")
            self.emit("movq %rax, %xmm0")
        elif k == TypeKind.DOUBLE:
            bits = struct.unpack("<Q", struct.pack("<d", v))[0]
            self.emit(f"movabs ${bits}, %rax")
            self.emit("movq %rax, %xmm0")
        else:
            raw = float_to_x87(v)
            lo = int.from_bytes(raw[:8], "little")
            hi = int.from_bytes(raw[8:10], "little")
            self.emit(f"movabs ${lo}, %rax")
            self.emit("mov %rax, -16(%rsp)")
            self.emit(f"movw ${hi}, -8(%rsp)")
            self.emit("fldt -16(%rsp)")

    def expr_VarRef(self, node) -> None:
        self.gen_addr(node)
        self.load(node.ty)

    def expr_Deref(self, node) -> None:
        self.gen_expr(node.operand)
        self.load(node.ty)

    def expr_AddrOf(self, node) -> None:
        self.gen_addr(node.operand)

    def expr_Member_(self, node) -> None:
        self.gen_addr(node)
        self.load(node.ty)
        m = node.member
        if m.is_bitfield:
            self.emit(f"shl ${64 - m.bit_width - m.bit_offset}, %rax")
            self.emit(f"{'shr' if m.ty.is_unsigned else 'sar'} ${64 - m.bit_width}, %rax")

    def expr_Assign(self, node) -> None:
        self.gen_addr(node.lhs)
        self.push()
        self.gen_expr(node.rhs)
        lhs = node.lhs
        if type(lhs).__name__ == "Member_" and lhs.member.is_bitfield:
            m = lhs.member
            self.emit("mov %rax, %r8")              # value to return
            self.emit("mov %rax, %rdi")
            mask = (1 << m.bit_width) - 1
            self.emit(f"movabs ${mask}, %rcx")
            self.emit("and %rcx, %rdi")
            self.emit(f"shl ${m.bit_offset}, %rdi")
            self.emit("mov (%rsp), %rax")
            self.load(m.ty)
            self.emit(f"movabs ${(~(mask << m.bit_offset)) & ((1 << 64) - 1)}, %r9")
            self.emit("and %r9, %rax")
            self.emit("or %rdi, %rax")
            self.store(node.ty)
            self.emit("mov %r8, %rax")
            # result: the assigned value truncated to the bitfield
            self.emit(f"shl ${64 - m.bit_width}, %rax")
            self.emit(f"{'shr' if m.ty.is_unsigned else 'sar'} ${64 - m.bit_width}, %rax")
            return
        self.store(node.ty)

    def expr_Comma(self, node) -> None:
        self.gen_expr(node.lhs)
        self.gen_expr(node.rhs)

    def expr_Cast(self, node) -> None:
        self.gen_expr(node.operand)
        self.cast(node.operand.ty, node.ty)

    def expr_MemZero(self, node) -> None:
        v = node.var
        self.emit(f"mov ${v.ty.size}, %rcx")
        self.emit(f"lea {v.offset}(%rbp), %rdi")
        self.emit("mov $0, %al")
        self.emit("rep stosb")

    def expr_Cond(self, node) -> None:
        els, end = self.new_label(), self.new_label()
        self.gen_expr(node.cond)
        self.cmp_zero(node.cond.ty)
        self.emit(f"je {els}")
        self.gen_expr(node.then)
        self.emit(f"jmp {end}")
        self.label(els)
        self.gen_expr(node.els)
        self.label(end)

    def expr_Unary(self, node) -> None:
        self.gen_expr(node.operand)
        ty = node.operand.ty
        if node.op == "!":
            self.cmp_zero(ty)
            self.emit("sete %al")
            self.emit("movzbl %al, %eax")
            return
        if node.op == "~":
            self.emit("not %rax")
            return
        k = node.ty.kind
        if k == TypeKind.FLOAT:
            self.emit("mov $1, %rax")
            self.emit("shl $31, %rax")
            self.emit("movq %rax, %xmm1")
            self.emit("xorps %xmm1, %xmm0")
        elif k == TypeKind.DOUBLE:
            self.emit("mov $1, %rax")
            self.emit("shl $63, %rax")
            self.emit("movq %rax, %xmm1")
            self.emit("xorpd %xmm1, %xmm0")
        elif k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
            self.emit("fchs")
        else:
            self.emit("neg %rax")

    def expr_Logical(self, node) -> None:
        short, end = self.new_label(), self.new_label()
        is_and = node.op == "&&"
        self.gen_expr(node.lhs)
        self.cmp_zero(node.lhs.ty)
        self.emit(f"{'je' if is_and else 'jne'} {short}")
        self.gen_expr(node.rhs)
        self.cmp_zero(node.rhs.ty)
        self.emit(f"{'je' if is_and else 'jne'} {short}")
        self.emit(f"mov ${1 if is_and else 0}, %rax")
        self.emit(f"jmp {end}")
        self.label(short)
        self.emit(f"mov ${0 if is_and else 1}, %rax")
        self.label(end)

    def expr_StmtExpr(self, node) -> None:
        for s in node.body.items:
            self.gen_stmt(s)

    def expr_LabelAddr(self, node) -> None:
        self.emit(f"lea {self.user_label(node.label)}(%rip), %rax")
