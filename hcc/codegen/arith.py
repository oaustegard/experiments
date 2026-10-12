"""Integer binary operators and comparisons (operands already converted to a
common type by sema). Floating-point operands are delegated to floats.py.

ArithMixin.expr_Binary(node): rhs -> push, lhs -> %rax, pop rhs -> %rdi.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass

from hcc.codegen.expressions import _contains_call
from hcc.sema.constexpr import try_eval_const
from hcc.typesys import TypeKind, is_decimal, is_flonum

_SETCC = {"==": ("sete", "sete"), "!=": ("setne", "setne"), "<": ("setl", "setb"),
          "<=": ("setle", "setbe"), ">": ("setg", "seta"), ">=": ("setge", "setae")}


def _contains_comma(node) -> bool:
    """True if the tree holds a Comma (e.g. a VLA length binding from vla_prefix)."""
    if type(node).__name__ == "Comma":
        return True
    if isinstance(node, list):
        return any(_contains_comma(x) for x in node)
    if not is_dataclass(node):
        return False
    return any(_contains_comma(getattr(node, f.name, None)) for f in fields(node)
               if f.name not in ("ty", "var", "member", "loc", "scope"))


# Functions that return twice: a longjmp re-enters them with the stack pointer
# they saved, so no operand may be parked across a call to one of them.
_RETURNS_TWICE = frozenset(("setjmp", "_setjmp", "sigsetjmp", "__sigsetjmp"))


def _is_returns_twice(node) -> bool:
    if type(node).__name__ == "BuiltinCall":
        return node.name == "setjmp"
    if type(node).__name__ != "Call":
        return False
    func = node.func
    while type(func).__name__ == "Cast":
        func = func.operand
    return type(func).__name__ == "VarRef" and func.var.name in _RETURNS_TWICE


def _contains_setjmp(node) -> bool:
    """True if the tree calls a returns-twice function (setjmp and kin)."""
    if _is_returns_twice(node):
        return True
    if isinstance(node, list):
        return any(_contains_setjmp(x) for x in node)
    if not is_dataclass(node):
        return False
    return any(_contains_setjmp(getattr(node, f.name, None)) for f in fields(node)
               if f.name not in ("ty", "var", "member", "loc", "scope"))


class ArithMixin:

    def expr_Binary(self, node) -> None:
        if node.result_buf is not None:
            return self.vector_binary(node)
        lty = node.lhs.ty
        if is_decimal(lty):
            return self.decimal_binary(node)
        if is_flonum(lty):
            return self.float_binary(node)
        if lty.kind == TypeKind.INT128:
            return self.int128_binary(node)
        # lhs-first when the lhs holds a VLA length binding (a Comma from
        # vla_prefix) that the rhs reads, e.g. `(*(int (*)[n])p)[1]`.
        rc = None if is_flonum(node.rhs.ty) or node.rhs.ty.size > 8 else try_eval_const(node.rhs)
        if rc is not None:
            # constant rhs: no operand needs parking on the stack
            self.gen_expr(node.lhs)
            self.emit(f"mov ${((rc + (1 << 63)) % (1 << 64)) - (1 << 63)}, %rdi")
        elif _contains_setjmp(node.rhs) and not _contains_setjmp(node.lhs):
            # Only the setjmp side may run with nothing parked: a value pushed
            # before the setjmp is popped after resumption, but later calls reuse
            # that slot. (selftest builtin_setjmp_operand_order.c)
            self.gen_expr(node.rhs)
            self.push()
            self.gen_expr(node.lhs)
            self.pop("%rdi")
        else:
            # Left-to-right, as gcc -O0 does: `f() - g()` calls f first, and
            # programs whose output depends on that order (e.g. doom's
            # `P_Random()-P_Random()`) must match the gcc reference.
            self.gen_expr(node.lhs)
            self.push()
            self.gen_expr(node.rhs)
            self.emit("mov %rax, %rdi")
            self.pop("%rax")
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
