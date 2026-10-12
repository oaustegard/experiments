"""__int128 / unsigned __int128 (16-byte integers).

A 128-bit integer value lives in the accumulator pair %rdx:%rax (%rdx = high
64 bits, %rax = low 64 bits). Memory access (memory.py) and literals and unary
operators (expressions.py) use the same convention.

Int128Mixin provides:
    is_i128(ty)              True for (unsigned) __int128
    int128_binary(node)      arithmetic, bitwise, shifts, comparisons
    int128_unary(op)         "~" or unary "-" on %rdx:%rax
    int128_cast(frm, to)     conversions touching 128-bit integers; returns
                             False when the generic cast table should be used
    int128_cmp_zero()        flags for "value != 0" (used by cmp_zero)
Division and float conversions call the libgcc helpers (__divti3, __fixdfti,
...) exactly as gcc-generated code does.
"""
from __future__ import annotations

from hcc.typesys import TypeKind, is_flonum
from hcc.codegen.casts import TABLE, type_id, I64, U64

_SETCC = {"==": "sete", "!=": "setne"}
# (signed, unsigned) set-instruction after a 128-bit subtraction
_ORDER = {"<": ("setl", "setb", False), ">=": ("setge", "setae", False),
          ">": ("setl", "setb", True), "<=": ("setge", "setae", True)}
_DIVMOD = {"/": "div", "%": "mod"}


def is_i128(ty) -> bool:
    return ty is not None and ty.kind == TypeKind.INT128


def _unsigned(ty) -> bool:
    return bool(ty.is_unsigned) or ty.kind in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.FUNC)


def _split(code):
    return [] if not code else code.split("; ")


class Int128Mixin:

    def _libgcc_call(self, name: str) -> None:
        # libgcc helpers need a 16-byte aligned %rsp at the call instruction;
        # the expression stack may be unaligned here, so align explicitly and
        # restore %rsp from the stack afterwards (r11 is not preserved by callees).
        self.emit("mov %rsp, %r11")
        self.emit("and $-16, %rsp")
        self.emit("sub $8, %rsp")
        self.emit("push %r11")
        self.emit(f"call {name}")
        self.emit("pop %rsp")

    def int128_binary(self, node) -> None:
        op = node.op
        lty = node.lhs.ty
        uns = _unsigned(lty)
        # rhs -> rdx:rax, pushed as two words; lhs -> rdx:rax; rhs popped to rsi:rdi
        self.gen_expr(node.rhs)
        self.emit("push %rdx")
        self.depth += 1
        self.push()
        self.gen_expr(node.lhs)
        self.pop("%rdi")                 # rhs low
        self.pop("%rsi")                 # rhs high
        if op == "+":
            self.emit("add %rdi, %rax")
            self.emit("adc %rsi, %rdx")
        elif op == "-":
            self.emit("sub %rdi, %rax")
            self.emit("sbb %rsi, %rdx")
        elif op == "*":
            # (ah:al)*(bh:bl) mod 2^128 = al*bl + ((ah*bl + al*bh) << 64)
            self.emit("mov %rax, %r8")
            self.emit("mov %rdx, %r9")
            self.emit("imul %rdi, %r9")
            self.emit("imul %rsi, %r8")
            self.emit("add %r8, %r9")
            self.emit("mul %rdi")
            self.emit("add %r9, %rdx")
        elif op in _DIVMOD:
            # helper args: rdi:rsi = dividend, rdx:rcx = divisor
            self.emit("mov %rdi, %r8")
            self.emit("mov %rsi, %rcx")
            self.emit("mov %rax, %rdi")
            self.emit("mov %rdx, %rsi")
            self.emit("mov %r8, %rdx")
            self._libgcc_call(f"__{'u' if uns else ''}{_DIVMOD[op]}ti3")
        elif op == "&":
            self.emit("and %rdi, %rax")
            self.emit("and %rsi, %rdx")
        elif op == "|":
            self.emit("or %rdi, %rax")
            self.emit("or %rsi, %rdx")
        elif op == "^":
            self.emit("xor %rdi, %rax")
            self.emit("xor %rsi, %rdx")
        elif op == "<<":
            self._shift_left()
        elif op == ">>":
            self._shift_right(lty.is_unsigned)
        elif op in _SETCC:
            self.emit("xor %rdi, %rax")
            self.emit("xor %rsi, %rdx")
            self.emit("or %rdx, %rax")
            self.emit(f"{_SETCC[op]} %al")
            self.emit("movzbl %al, %eax")
        elif op in _ORDER:
            signed_set, unsigned_set, swap = _ORDER[op]
            if swap:
                self.emit("cmp %rax, %rdi")
                self.emit("sbb %rdx, %rsi")
            else:
                self.emit("cmp %rdi, %rax")
                self.emit("sbb %rsi, %rdx")
            self.emit(f"{unsigned_set if uns else signed_set} %al")
            self.emit("movzbl %al, %eax")
        else:
            self.error_unsupported(node)

    def _shift_left(self) -> None:
        skip = self.new_label()
        self.emit("mov %rdi, %rcx")
        self.emit("shld %cl, %rax, %rdx")
        self.emit("shl %cl, %rax")
        self.emit("test $64, %cl")
        self.emit(f"je {skip}")
        self.emit("mov %rax, %rdx")
        self.emit("xor %eax, %eax")
        self.label(skip)

    def _shift_right(self, unsigned: bool) -> None:
        skip = self.new_label()
        self.emit("mov %rdi, %rcx")
        self.emit("shrd %cl, %rdx, %rax")
        if unsigned:
            self.emit("shr %cl, %rdx")
        else:
            self.emit("sar %cl, %rdx")
        self.emit("test $64, %cl")
        self.emit(f"je {skip}")
        if unsigned:
            self.emit("mov %rdx, %rax")
            self.emit("xor %edx, %edx")
        else:
            self.emit("mov %rdx, %rax")
            self.emit("sar $63, %rdx")
        self.label(skip)

    def int128_unary(self, op: str) -> None:
        if op == "~":
            self.emit("not %rax")
            self.emit("not %rdx")
        else:                            # two's complement negate of rdx:rax
            self.emit("neg %rax")
            self.emit("adc $0, %rdx")
            self.emit("neg %rdx")

    def int128_cmp_zero(self) -> None:
        self.emit("mov %rax, %rcx")
        self.emit("or %rdx, %rcx")
        self.emit("cmp $0, %rcx")

    def int128_cast(self, frm, to) -> bool:
        if is_i128(to):
            if is_flonum(frm):
                if frm.kind not in (TypeKind.FLOAT, TypeKind.DOUBLE):
                    return False
                sfx = "sf" if frm.kind == TypeKind.FLOAT else "df"
                self._libgcc_call(f"__fix{'uns' if _unsigned(to) else ''}{sfx}ti")
                return True
            if is_i128(frm):
                return True
            uns = _unsigned(frm)
            for part in _split(TABLE[type_id(frm)][U64 if uns else I64]):
                self.emit(part)
            if uns:
                self.emit("xor %edx, %edx")
            else:
                self.emit("mov %rax, %rdx")
                self.emit("sar $63, %rdx")
            return True
        if is_i128(frm) and is_flonum(to) and to.kind in (TypeKind.FLOAT, TypeKind.DOUBLE,
                                                          TypeKind.LDOUBLE):
            self.emit("mov %rax, %rdi")
            self.emit("mov %rdx, %rsi")
            sfx = {TypeKind.FLOAT: "sf", TypeKind.DOUBLE: "df", TypeKind.LDOUBLE: "xf"}[to.kind]
            self._libgcc_call(f"__float{'un' if _unsigned(frm) else ''}ti{sfx}")
            return True
        return False
