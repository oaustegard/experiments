"""Scalar `__int128` / `unsigned __int128` values (TypeKind.INT128).

A value lives in %rdx:%rax (low 64 bits in %rax, high in %rdx); this is the
accumulator convention for 16-byte integers, the scalar counterpart of the
16-byte lanes in vector_wide.py. Covered here: + - * / % & | ^ << >>,
comparisons, unary - ~ !, and conversions to/from integers and float/double.
Division and remainder call the libgcc helpers (__divti3, __udivti3, ...),
which the gcc driver links anyway. Loads/stores live in memory.py.
"""
from __future__ import annotations

from hcc import diag
from hcc.codegen.expressions import _contains_call
from hcc.codegen.vector_wide import _wide_shift
from hcc.typesys import TY_INT128, TY_LONG, TY_ULONG, TypeKind

_DIVMOD_HELPER = {
    ("/", False): "__udivti3",
    ("/", True): "__divti3",
    ("%", False): "__umodti3",
    ("%", True): "__modti3",
}
# (signed, unsigned) condition codes for `a OP b` computed as a - b (or b - a)
_ORDER_CC = {"<": ("l", "b"), ">": ("l", "b"), "<=": ("ge", "ae"), ">=": ("ge", "ae")}
_BITWISE = {"&": "and", "|": "or", "^": "xor"}
_SHIFTS = ("<<", ">>")
_COMPARES = ("==", "!=", "<", "<=", ">", ">=")


def is_int128(ty) -> bool:
    return ty is not None and ty.kind == TypeKind.INT128


def _is_unsigned_src(ty) -> bool:
    return bool(ty.is_unsigned) or ty.kind in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.FUNC)


class Int128Mixin:

    def gen_int128(self, e) -> None:
        """Evaluate e into %rdx:%rax, widening non-__int128 operands."""
        self.gen_expr(e)
        if not is_int128(e.ty):
            self.cast(e.ty, TY_INT128)

    def int128_binary(self, node) -> None:
        """Binary node whose lhs (or, for comparisons, operands) is __int128."""
        op = node.op
        signed = not node.lhs.ty.is_unsigned
        if op in _SHIFTS:
            # the count is a 64-bit integer (its low 64 bits are used, mod 128)
            self.gen_expr(node.rhs)
            self.cast(node.rhs.ty, TY_ULONG)
            self.push()
            self.gen_int128(node.lhs)
            self.pop("%rcx")
            _wide_shift(self, op, signed)
            return
        if _contains_call(node.lhs) and not _contains_call(node.rhs):
            # park the call-free rhs-side last; lhs first (see arith.py)
            self.gen_int128(node.lhs)
            self.emit("push %rdx")
            self.emit("push %rax")
            self.depth += 2
            self.gen_int128(node.rhs)
            self.emit("mov %rax, %rcx")
            self.emit("mov %rdx, %rsi")
            self.pop("%rax")
            self.pop("%rdx")
        else:
            self.gen_int128(node.rhs)
            self.emit("push %rdx")
            self.emit("push %rax")
            self.depth += 2
            self.gen_int128(node.lhs)
            self.pop("%rcx")                 # rhs low
            self.pop("%rsi")                 # rhs high
        # lhs = %rdx:%rax, rhs = %rsi:%rcx from here on
        if op in ("+", "-"):
            self.emit(f"{'add' if op == '+' else 'sub'} %rcx, %rax")
            self.emit(f"{'adc' if op == '+' else 'sbb'} %rsi, %rdx")
        elif op in _BITWISE:
            self.emit(f"{_BITWISE[op]} %rcx, %rax")
            self.emit(f"{_BITWISE[op]} %rsi, %rdx")
        elif op == "*":
            self.int128_mul()
        elif op in ("/", "%"):
            self.int128_divmod(op, signed)
        elif op in _COMPARES:
            self.int128_compare(op, signed)
        else:
            diag.error(node, f"codegen: operator '{op}' on __int128 is not supported", "codegen")

    def int128_mul(self) -> None:
        """Low 128 bits of the product: lo*lo full, plus the two cross terms' low halves."""
        self.emit("mov %rdx, %r8")
        self.emit("imul %rcx, %r8")          # ahi * blo
        self.emit("mov %rax, %r9")
        self.emit("imul %rsi, %r9")          # alo * bhi
        self.emit("add %r9, %r8")
        self.emit("mul %rcx")                # rdx:rax = alo * blo
        self.emit("add %r8, %rdx")

    def int128_divmod(self, op: str, signed: bool) -> None:
        """Call the libgcc helper: rdi:rsi = dividend, rdx:rcx = divisor."""
        self.emit("mov %rdx, %r8")           # a_hi
        self.emit("mov %rsi, %r9")           # b_hi
        self.emit("mov %rax, %rdi")
        self.emit("mov %r8, %rsi")
        self.emit("mov %rcx, %rdx")          # b_lo
        self.emit("mov %r9, %rcx")           # b_hi
        self.libcall(_DIVMOD_HELPER[(op, signed)])

    def libcall(self, name: str) -> None:
        """Call a runtime helper with %rsp 16-byte aligned (depth must be even)."""
        pad = 8 if self.depth % 2 else 0
        if pad:
            self.emit("sub $8, %rsp")
        self.emit(f"call {name}")
        if pad:
            self.emit("add $8, %rsp")

    def int128_compare(self, op: str, signed: bool) -> None:
        """Leaves the 0/1 result (int) in %eax."""
        if op in ("==", "!="):
            self.emit("cmp %rcx, %rax")
            self.emit("sete %r8b")
            self.emit("cmp %rsi, %rdx")
            self.emit("sete %r9b")
            self.emit("and %r9b, %r8b")
            if op == "!=":
                self.emit("xor $1, %r8b")
            self.emit("movzbl %r8b, %eax")
            return
        # 128-bit subtraction with borrow; a > b and a <= b use b - a instead.
        if op in (">", "<="):
            self.emit("cmp %rax, %rcx")
            self.emit("sbb %rdx, %rsi")
        else:
            self.emit("cmp %rcx, %rax")
            self.emit("sbb %rsi, %rdx")
        cc = _ORDER_CC[op][0 if signed else 1]
        self.emit(f"set{cc} %al")
        self.emit("movzbl %al, %eax")

    def int128_unary(self, op: str) -> None:
        """Operand already in %rdx:%rax."""
        if op == "-":
            self.emit("not %rax")
            self.emit("not %rdx")
            self.emit("add $1, %rax")
            self.emit("adc $0, %rdx")
        elif op == "~":
            self.emit("not %rax")
            self.emit("not %rdx")
        elif op == "!":
            self.emit("or %rdx, %rax")
            self.emit("sete %al")
            self.emit("movzbl %al, %eax")
        else:
            diag.error(None, f"codegen: unary '{op}' on __int128 is not supported", "codegen")

    def int128_cast(self, frm, to) -> None:
        """Conversions where either side is __int128 (value in %rdx:%rax)."""
        if is_int128(to):
            if is_int128(frm):
                return
            if frm.kind in (TypeKind.FLOAT, TypeKind.DOUBLE):
                suffix = "sf" if frm.kind == TypeKind.FLOAT else "df"
                unsigned = "uns" if to.is_unsigned else ""
                self.libcall(f"__fix{unsigned}{suffix}ti")
                return
            if frm.size < 8:
                self.cast(frm, TY_ULONG if _is_unsigned_src(frm) else TY_LONG)
            if _is_unsigned_src(frm):
                self.emit("xor %edx, %edx")
            else:
                self.emit("mov %rax, %rdx")
                self.emit("sar $63, %rdx")
            return
        # __int128 -> something else
        if to.kind == TypeKind.BOOL:
            self.emit("or %rdx, %rax")
            self.emit("setne %al")
            self.emit("movzbl %al, %eax")
            return
        if to.kind in (TypeKind.FLOAT, TypeKind.DOUBLE):
            self.emit("mov %rax, %rdi")
            self.emit("mov %rdx, %rsi")
            suffix = "sf" if to.kind == TypeKind.FLOAT else "df"
            self.libcall(f"__float{'un' if frm.is_unsigned else ''}ti{suffix}")
            return
        if to.kind in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
            diag.error(None, "codegen: conversion of __int128 to long double is not supported",
                       "codegen")
        self.cast(TY_ULONG, to)              # the low 64 bits, then the usual narrowing
