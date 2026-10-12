"""Floating-point binary operators: float/double via SSE, long double via x87.

FloatMixin.float_binary(node) is called by arith.expr_Binary.
SSE: rhs -> pushf, lhs -> %xmm0, popf -> %xmm1.  x87: lhs then rhs on the
FPU stack (st1 = lhs, st0 = rhs).
"""
from __future__ import annotations

from hcc.typesys import TypeKind


class FloatMixin:

    def float_binary(self, node) -> None:
        k = node.lhs.ty.kind
        if k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
            return self.x87_binary(node)
        sz = "ss" if k == TypeKind.FLOAT else "sd"
        self.gen_expr(node.rhs)
        self.pushf()
        self.gen_expr(node.lhs)
        self.popf(1)
        op = node.op
        if op in ("+", "-", "*", "/"):
            ins = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[op]
            self.emit(f"{ins}{sz} %xmm1, %xmm0")
            return
        if op == "==":
            self.emit(f"ucomi{sz} %xmm1, %xmm0")
            self.emit("sete %al")
            self.emit("setnp %dl")
            self.emit("and %dl, %al")
        elif op == "!=":
            self.emit(f"ucomi{sz} %xmm1, %xmm0")
            self.emit("setne %al")
            self.emit("setp %dl")
            self.emit("or %dl, %al")
        elif op == "<":
            self.emit(f"ucomi{sz} %xmm0, %xmm1")
            self.emit("seta %al")
        elif op == "<=":
            self.emit(f"ucomi{sz} %xmm0, %xmm1")
            self.emit("setae %al")
        elif op == ">":
            self.emit(f"ucomi{sz} %xmm1, %xmm0")
            self.emit("seta %al")
        elif op == ">=":
            self.emit(f"ucomi{sz} %xmm1, %xmm0")
            self.emit("setae %al")
        else:
            self.error_unsupported(node)
        self.emit("and $1, %al")
        self.emit("movzbl %al, %eax")

    def x87_binary(self, node) -> None:
        self.gen_expr(node.lhs)
        self.gen_expr(node.rhs)
        op = node.op
        if op in ("+", "-", "*", "/"):
            self.emit({"+": "faddp", "-": "fsubrp", "*": "fmulp", "/": "fdivrp"}[op])
            return
        self.emit("fcomip")
        self.emit("fstp %st(0)")
        cc = {"==": "sete", "!=": "setne", "<": "seta", "<=": "setae", ">": "setb", ">=": "setbe"}[op]
        self.emit(f"{cc} %al")
        self.emit("movzbl %al, %eax")
