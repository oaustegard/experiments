"""Floating-point binary operators: float/double via SSE, long double via x87.

FloatMixin.float_binary(node) is called by arith.expr_Binary.
SSE: rhs -> pushf, lhs -> %xmm0, popf -> %xmm1.  x87: lhs then rhs on the
FPU stack (st1 = lhs, st0 = rhs).
"""
from __future__ import annotations

from hcc.codegen.half import ROUND_HALF
from hcc.typesys import TypeKind


class FloatMixin:

    def float_binary(self, node) -> None:
        k = node.lhs.ty.kind
        if k == TypeKind.FLOAT128:
            return self.f128_binary(node)
        if k == TypeKind.LDOUBLE:
            return self.x87_binary(node)
        sz = "ss" if k in (TypeKind.FLOAT16, TypeKind.FLOAT) else "sd"
        self.gen_expr(node.rhs)
        self.pushf()
        self.gen_expr(node.lhs)
        self.popf(1)
        op = node.op
        if op in ("+", "-", "*", "/"):
            ins = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[op]
            self.emit(f"{ins}{sz} %xmm1, %xmm0")
            if node.ty.kind == TypeKind.FLOAT16:
                self.emit(ROUND_HALF)       # half arithmetic: round the float result
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
        # Ordered relations use comis: C requires FE_INVALID for any NaN here.
        elif op == "<":
            self.emit(f"comi{sz} %xmm0, %xmm1")
            self.emit("seta %al")
        elif op == "<=":
            self.emit(f"comi{sz} %xmm0, %xmm1")
            self.emit("setae %al")
        elif op == ">":
            self.emit(f"comi{sz} %xmm1, %xmm0")
            self.emit("seta %al")
        elif op == ">=":
            self.emit(f"comi{sz} %xmm1, %xmm0")
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
        if op in ("==", "!="):  # quiet compare; unordered sets ZF and PF
            self.emit("fucomip %st(1), %st")
            self.emit("fstp %st(0)")
            if op == "==":
                self.emit("sete %al")
                self.emit("setnp %dl")
                self.emit("and %dl, %al")
            else:
                self.emit("setne %al")
                self.emit("setp %dl")
                self.emit("or %dl, %al")
        else:
            # Ordered relations: fcomip sets CF=1 for unordered (NaN) operands, so
            # setb/setbe would be true there. '>' and '>=' swap the operands and use
            # seta/setae, which are false on unordered (CF=1), as C requires.
            if op in (">", ">="):
                self.emit("fxch %st(1)")             # st0 = lhs, st1 = rhs
            self.emit("fcomip")
            self.emit("fstp %st(0)")
            cc = {"<": "seta", "<=": "setae", ">": "seta", ">=": "setae"}[op]
            self.emit(f"{cc} %al")
        self.emit("movzbl %al, %eax")

    def fp_cmp_builtin(self, node) -> None:
        """__builtin_is<op>(a, b) (C99 quiet comparisons): ucomi/fucomip set
        ZF,PF,CF = 111 when unordered, so every relation except isunordered
        and islessgreater is false for NaN. 'less' forms compare b against a
        so that seta/setae cannot be fooled by CF=1 on unordered operands."""
        name = node.name
        swap = name in ("isless", "islessequal")
        lhs, rhs = node.args
        if lhs.ty.kind == TypeKind.FLOAT128:
            return self.f128_fp_cmp(name, lhs, rhs)
        if lhs.ty.kind == TypeKind.LDOUBLE:
            self.gen_expr(lhs)
            self.gen_expr(rhs)                       # st0 = rhs, st1 = lhs
            if not swap:
                self.emit("fxch %st(1)")             # st0 = lhs, st1 = rhs
            self.emit("fucomip %st(1), %st")
            self.emit("fstp %st(0)")
        else:
            self.gen_expr(rhs)
            self.pushf()
            self.gen_expr(lhs)
            self.popf(1)                             # xmm0 = lhs, xmm1 = rhs
            self.emit("ucomisd %xmm0, %xmm1" if swap else "ucomisd %xmm1, %xmm0")
        cc = {"isunordered": "setp", "isgreater": "seta", "isgreaterequal": "setae",
              "isless": "seta", "islessequal": "setae", "islessgreater": "setne"}[name]
        self.emit(f"{cc} %al")
        self.emit("and $1, %al")
        self.emit("movzbl %al, %eax")
