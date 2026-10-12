"""Integer binary operators and comparisons (operands already converted to a
common type by sema). Floating-point operands are delegated to floats.py.

ArithMixin.expr_Binary(node): rhs -> push, lhs -> %rax, pop rhs -> %rdi.
"""
from __future__ import annotations

from hcc.codegen.expressions import _contains_call
from hcc.typesys import TypeKind, is_flonum

_SETCC = {"==": ("sete", "sete"), "!=": ("setne", "setne"), "<": ("setl", "setb"),
          "<=": ("setle", "setbe"), ">": ("setg", "seta"), ">=": ("setge", "setae")}


class ArithMixin:

    def expr_Binary(self, node) -> None:
        if node.result_buf is not None:
            return self.vector_binary(node)
        lty = node.lhs.ty
        if is_flonum(lty):
            return self.float_binary(node)
        if _contains_call(node.lhs) and not _contains_call(node.rhs):
            # Only a call-free operand may be parked on the stack: a value
            # pushed before a call (e.g. setjmp, whose resumption reuses the
            # stack slot) would be clobbered. Evaluate the call side first.
            self.gen_expr(node.lhs)
            self.push()
            self.gen_expr(node.rhs)
            self.emit("mov %rax, %rdi")
            self.pop("%rax")
        else:
            self.gen_expr(node.rhs)
            self.push()
            self.gen_expr(node.lhs)
            self.pop("%rdi")
        wide = lty.size == 8 or lty.kind in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.FUNC) or \
            lty.size > 8
        ax, di, dx = ("%rax", "%rdi", "%rdx") if wide else ("%eax", "%edi", "%edx")
        op = node.op
        uns = lty.is_unsigned or lty.kind in (TypeKind.PTR, TypeKind.ARRAY)
        if op == "+":
            self.emit(f"add {di}, {ax}")
        elif op == "-":
            self.emit(f"sub {di}, {ax}")
        elif op == "*":
            self.emit(f"imul {di}, {ax}")
        elif op in ("/", "%"):
            if uns:
                self.emit("mov $0, %edx")
                self.emit(f"div {di}")
            else:
                self.emit("cqo" if wide else "cdq")
                self.emit(f"idiv {di}")
            if op == "%":
                self.emit(f"mov {dx}, {ax}")
        elif op == "&":
            self.emit(f"and {di}, {ax}")
        elif op == "|":
            self.emit(f"or {di}, {ax}")
        elif op == "^":
            self.emit(f"xor {di}, {ax}")
        elif op == "<<":
            self.emit("mov %rdi, %rcx")
            self.emit(f"shl %cl, {ax}")
            if lty.size == 16:
                # hcc keeps 128-bit values in one 64-bit word: a count of 64 or
                # more must clear it (x86 would wrap the count modulo 64).
                self.emit("xor %edx, %edx")
                self.emit("cmp $64, %rcx")
                self.emit("cmovae %rdx, %rax")
        elif op == ">>":
            self.emit("mov %rdi, %rcx")
            self.emit(f"{'shr' if lty.is_unsigned else 'sar'} %cl, {ax}")
        elif op in _SETCC:
            self.emit(f"cmp {di}, {ax}")
            self.emit(_SETCC[op][1 if uns else 0] + " %al")
            self.emit("movzbl %al, %eax")
        else:
            self.error_unsupported(node)

    def error_unsupported(self, node):
        from hcc import diag
        diag.error(node, f"codegen: unsupported operator '{node.op}'", "codegen")
