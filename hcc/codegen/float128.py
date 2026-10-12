"""_Float128 (IEEE binary128) codegen.

Register model: a _Float128 value lives in all of %xmm0 (the SysV class SSE+SSEUP,
so arguments and results use one xmm register). Arithmetic, comparisons and
conversions call libgcc's soft-float routines (__addtf3, __eqtf2, __fixtfsi, ...),
which the gcc link driver always pulls in. Helpers for loads/stores/literals are
here too; memory.py, floats.py, casts.py and calls.py dispatch to them.
"""
from __future__ import annotations

from hcc import diag
from hcc.f128_lit import float128_bits
from hcc.typesys import TypeKind

_ARITH = {"+": "__addtf3", "-": "__subtf3", "*": "__multf3", "/": "__divtf3"}
# op -> (libgcc comparison, condition code on its int result)
_CMP = {"==": ("__eqtf2", "sete"), "!=": ("__netf2", "setne"),
        "<": ("__lttf2", "setl"), "<=": ("__letf2", "setle"),
        ">": ("__gttf2", "setg"), ">=": ("__getf2", "setge")}


class Float128Mixin:

    def f128_libcall(self, name: str) -> None:
        """Call a libgcc helper with 16-byte stack alignment preserved."""
        if self.depth % 2:
            self.emit("sub $8, %rsp")
            self.emit(f"call {name}")
            self.emit("add $8, %rsp")
        else:
            self.emit(f"call {name}")

    def f128_push(self) -> None:
        """Save the xmm0 value on the machine stack (16 bytes)."""
        self.emit("sub $16, %rsp")
        self.emit("movdqu %xmm0, (%rsp)")
        self.depth += 2

    def f128_pop(self, n: int) -> None:
        self.emit(f"movdqu (%rsp), %xmm{n}")
        self.emit("add $16, %rsp")
        self.depth -= 2

    def f128_load(self) -> None:
        self.emit("movdqu (%rax), %xmm0")

    def f128_store(self) -> None:
        self.emit("movdqu %xmm0, (%rdi)")

    def f128_neg(self) -> None:
        self.emit("pcmpeqd %xmm1, %xmm1")
        self.emit("psllq $63, %xmm1")
        self.emit("pslldq $8, %xmm1")        # only the sign bit of the high quadword
        self.emit("xorps %xmm1, %xmm0")

    def f128_binary(self, node) -> None:
        op = node.op
        self.gen_expr(node.rhs)
        self.f128_push()
        self.gen_expr(node.lhs)
        self.f128_pop(1)                     # xmm0 = lhs, xmm1 = rhs
        if op in _ARITH:
            self.f128_libcall(_ARITH[op])
            return
        if op not in _CMP:
            self.error_unsupported(node)
        fn, cc = _CMP[op]
        self.f128_libcall(fn)                # int result: sign/zero encodes the relation
        self.emit("test %eax, %eax")
        self.emit(f"{cc} %al")
        self.emit("movzbl %al, %eax")

    def f128_literal(self, node) -> None:
        bits = node.bits if node.bits is not None else float128_bits(float(node.value))
        bits &= (1 << 128) - 1
        self.emit(f"movabs ${bits & ((1 << 64) - 1)}, %rax")
        self.emit("mov %rax, -16(%rsp)")
        self.emit(f"movabs ${bits >> 64}, %rax")
        self.emit("mov %rax, -8(%rsp)")
        self.emit("movdqu -16(%rsp), %xmm0")

    def f128_is_zero(self) -> None:
        """Sets ZF when the value is +0 or -0 (clears the sign bit first)."""
        self.emit("movdqu %xmm0, -16(%rsp)")
        self.emit("mov -8(%rsp), %r11")
        self.emit("add %r11, %r11")          # drop the sign bit; ZF from the high half
        self.emit("or -16(%rsp), %r11")

    def f128_cast(self, frm, to) -> bool:
        """Convert the accumulator between _Float128 and another scalar type.
        Returns False when the caller should use the generic table."""
        fk, tk = frm.kind, to.kind
        if fk == tk:
            return True
        if fk == TypeKind.FLOAT128:
            if tk == TypeKind.FLOAT or tk == TypeKind.FLOAT16:
                self.f128_libcall("__trunctfsf2")
            elif tk == TypeKind.DOUBLE:
                self.f128_libcall("__trunctfdf2")
            elif tk == TypeKind.LDOUBLE:
                self.f128_libcall("__trunctfxf2")   # st(0) holds the result
            elif tk == TypeKind.INT128:          # result: %rdx:%rax = high:low
                self.f128_libcall("__fixunstfti" if to.is_unsigned else "__fixtfti")
            elif tk == TypeKind.BOOL:
                self.f128_is_zero()
                self.emit("setne %al")
                self.emit("movzbl %al, %eax")
            elif to.size == 8 and to.is_unsigned:
                self.f128_libcall("__fixunstfdi")
            elif to.size == 8 or (to.size == 4 and to.is_unsigned):
                self.f128_libcall("__fixtfdi")
                if to.size == 4:
                    self.emit("mov %eax, %eax")
            else:
                self.f128_libcall("__fixtfsi")
                self.emit({1: "movsbl %al, %eax" if not to.is_unsigned else "movzbl %al, %eax",
                           2: "movswl %ax, %eax" if not to.is_unsigned else "movzwl %ax, %eax",
                           4: ""}[to.size])
            return True
        # some scalar -> _Float128
        if fk == TypeKind.FLOAT or fk == TypeKind.FLOAT16:
            self.f128_libcall("__extendsftf2")
        elif fk == TypeKind.DOUBLE:
            self.f128_libcall("__extenddftf2")
        elif fk == TypeKind.LDOUBLE:
            # long double is passed in memory: the 16-byte slot must sit at the
            # call's %rsp, so any alignment padding goes below it.
            pad = 8 if self.depth % 2 else 0
            if pad:
                self.emit("sub $8, %rsp")
            self.emit("sub $16, %rsp")
            self.emit("fstpt (%rsp)")
            self.emit("call __extendxftf2")
            self.emit(f"add ${16 + pad}, %rsp")
        elif fk == TypeKind.INT128:          # argument: low word in %rdi, high in %rsi
            self.emit("mov %rdx, %rsi")
            self.emit("mov %rax, %rdi")
            self.f128_libcall("__floatuntitf" if frm.is_unsigned else "__floattitf")
        elif fk in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.FUNC, TypeKind.STRUCT, TypeKind.UNION):
            diag.error(frm, "invalid conversion to _Float128", "codegen")
        else:
            self.f128_widen_int(frm)
            if frm.is_unsigned and frm.size == 8:
                self.f128_libcall("__floatunditf")
            else:
                self.f128_libcall("__floatditf")
        return True

    def f128_widen_int(self, frm) -> None:
        """Extend the integer in %eax/%rax to 64 bits by its own signedness."""
        sz, uns = frm.size, frm.is_unsigned
        if sz == 1:
            self.emit("movzbl %al, %eax" if uns else "movsbq %al, %rax")
        elif sz == 2:
            self.emit("movzwl %ax, %eax" if uns else "movswq %ax, %rax")
        elif sz == 4:
            self.emit("mov %eax, %eax" if uns else "movslq %eax, %rax")
        self.emit("mov %rax, %rdi")

    def f128_fp_cmp(self, name: str, lhs, rhs) -> None:
        """__builtin_is<rel>(a, b) on _Float128 operands: quiet relations. libgcc's
        ordered compares return a value that cannot satisfy the test for NaN
        (lt/le -> 1, gt/ge -> -1), so the signed result tests are exact."""
        self.gen_expr(rhs)
        self.f128_push()
        self.gen_expr(lhs)
        self.f128_pop(1)                     # xmm0 = lhs, xmm1 = rhs
        if name == "islessgreater":
            # (a < b) || (a > b): both operands must survive the first call
            self.emit("sub $48, %rsp")
            self.emit("movdqu %xmm0, (%rsp)")
            self.emit("movdqu %xmm1, 16(%rsp)")
            self.depth += 6
            self.f128_libcall("__lttf2")
            self.emit("mov %eax, 32(%rsp)")
            self.emit("movdqu (%rsp), %xmm0")
            self.emit("movdqu 16(%rsp), %xmm1")
            self.f128_libcall("__gttf2")
            self.emit("cmp $0, %eax")
            self.emit("setg %dl")
            self.emit("cmpl $0, 32(%rsp)")
            self.emit("setl %cl")
            self.emit("or %cl, %dl")
            self.emit("movzbl %dl, %eax")
            self.emit("add $48, %rsp")
            self.depth -= 6
            return
        if name == "isunordered":
            self.f128_libcall("__unordtf2")
            self.emit("test %eax, %eax")
            cc = "setne"
        else:
            fn, cc = {"isless": ("__lttf2", "setl"), "islessequal": ("__letf2", "setle"),
                      "isgreater": ("__gttf2", "setg"), "isgreaterequal": ("__getf2", "setge")}[name]
            self.f128_libcall(fn)
            self.emit("cmp $0, %eax")
        self.emit(f"{cc} %al")
        self.emit("movzbl %al, %eax")
