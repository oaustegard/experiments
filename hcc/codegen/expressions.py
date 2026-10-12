"""Expression codegen: literals, variables, unary ops, assignment, ?:, comma,
logical ops, casts, member access (incl. bitfields), statement expressions.

ExpressionMixin: one `expr_<NodeClass>` method per node. Binary arithmetic is
in arith.py (integers) and floats.py; calls in calls.py; builtins in builtins.py.
"""
from __future__ import annotations

import struct

from hcc.sema.constexpr import try_eval_const
from hcc.sema.initializers import float_to_x87
from dataclasses import fields, is_dataclass

from hcc.typesys import TypeKind, is_flonum, is_integer


def _contains_call(node) -> bool:
    """True if the expression tree under `node` contains a function call."""
    if type(node).__name__ == "Call":
        return True
    if isinstance(node, list):
        return any(_contains_call(x) for x in node)
    if not is_dataclass(node):
        return False
    return any(_contains_call(getattr(node, f.name, None)) for f in fields(node)
               if f.name not in ("ty", "var", "member", "loc", "scope"))


def _calls_before_store(node) -> bool:
    """Scalar integer/pointer assignment whose rhs contains a call and whose
    lhs is not a bitfield: the rhs may be evaluated before the lhs address."""
    lhs, ty = node.lhs, node.ty
    if ty is None or not (is_integer(ty) or ty.kind == TypeKind.PTR) or ty.size > 8:
        return False
    if type(lhs).__name__ == "Member_" and lhs.member.is_bitfield:
        return False
    return _contains_call(node.rhs)


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
        if _calls_before_store(node):
            # Evaluate a call-containing scalar rhs first so no lhs address
            # temp is live across the call (setjmp/longjmp re-entry would
            # otherwise find the saved address clobbered).
            self.gen_expr(node.rhs)
            self.push()                     # value
            self.gen_addr(node.lhs)
            self.push()                     # address (popped by store)
            self.emit("mov 8(%rsp), %rax")  # value back into %rax
            self.store(node.ty)
            self.emit("add $8, %rsp")       # drop the saved value slot
            self.depth -= 1
            return
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
        self.discard_value(node.lhs.ty)
        self.gen_expr(node.rhs)

    def expr_Cast(self, node) -> None:
        if is_flonum(node.operand.ty) and is_integer(node.ty):
            # constant float->int conversions are folded (saturating, like GCC)
            folded = try_eval_const(node)
            if folded is not None:
                self.emit(f"movabs ${folded}, %rax")
                return
        self.gen_expr(node.operand)
        if node.operand.ty.is_vector:
            # vector values are addresses: reinterpret the same bytes
            if not node.ty.is_vector:
                self.load(node.ty)
            return
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
        if node.result_buf is not None:
            if node.op == "-":
                self.vector_neg(node)
            else:
                self.vector_not(node)
            return
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

    def vector_not(self, node) -> None:
        """`~v` on a GCC vector: copy v into the result buffer, then invert it
        word by word. Leaves %rax = address of the result (like struct values)."""
        self.gen_expr(node.operand)                  # %rax = source address
        buf = node.result_buf
        size = node.operand.ty.size
        self.emit(f"lea {buf.offset}(%rbp), %rdi")
        self.copy_struct(size)                       # %rdi = dest, %rax = src
        i = 0
        while i + 8 <= size:
            self.emit(f"notq {i}(%rdi)")
            i += 8
        while i + 4 <= size:
            self.emit(f"notl {i}(%rdi)")
            i += 4
        while i < size:
            self.emit(f"notb {i}(%rdi)")
            i += 1
        self.emit("mov %rdi, %rax")
