"""Bit-level floating classification builtins (C99 <math.h> type-generic macros).

  __builtin_fpclassify(nan, inf, normal, subnormal, zero, x) -> one of the five
  constants (the first five args are integer constants, see hcc/builtins/fpclass.py)
  __builtin_isnormal(x)      -> 1 for a normal finite value, else 0
  __builtin_isinf_sign(x)    -> -1 / 1 for -inf / +inf, else 0

The value is classified from its raw bits, so NaN/inf/subnormal/zero are exact
for float, double, _Float16, and the x87 format (long double, _Float128).
Register contract of `_classify`: input value in %xmm0 (float/double/_Float16)
or st(0) (x87); output category number in %eax, sign bit in %r8d.
"""

from __future__ import annotations

from hcc import diag
from hcc.typesys import TypeKind

# category numbers produced in %eax, in the order of the five fpclassify constants
_NAN, _INF, _NORMAL, _SUB, _ZERO = range(5)
# smallest positive normal binary16 (2^-14) as float bits; _Float16 values are
# held as floats, so a half subnormal is a float with exponent below this
_HALF_MIN_NORMAL_BITS = 0x38800000


class FpClassifyMixin:

    def builtin_fpclassify(self, node) -> None:
        consts = [a.value & 0xFFFFFFFF for a in node.args[:5]]
        self._classify(node.args[5])
        # select the constant for the category in %eax (cmov chain)
        self.emit("mov $0, %ecx")
        for cat in range(5):
            self.emit(f"mov ${consts[cat]}, %edx")
            self.emit(f"cmp ${cat}, %eax")
            self.emit("cmove %edx, %ecx")
        self.emit("mov %ecx, %eax")

    def builtin_isnormal(self, node) -> None:
        self._classify(node.args[0])
        self.emit(f"cmp ${_NORMAL}, %eax")
        self.emit("sete %al")
        self.emit("movzbl %al, %eax")

    def builtin_isnan(self, node) -> None:
        self._classify_flag(node.args[0], "e", _NAN)

    def builtin_isinf(self, node) -> None:
        self._classify_flag(node.args[0], "e", _INF)

    def builtin_isfinite(self, node) -> None:
        # zero, subnormal and normal are the categories 2..4 (see _NAN.._ZERO order)
        self._classify_flag(node.args[0], "ae", _NORMAL)

    def _classify_flag(self, x, cond: str, cat: int) -> None:
        """0/1 result: category == cat ("e") or category >= cat ("ae") in %eax."""
        self._classify(x)
        self.emit(f"cmp ${cat}, %eax")
        self.emit("sete %al" if cond == "e" else "setae %al")
        self.emit("movzbl %al, %eax")

    def builtin_isinf_sign(self, node) -> None:
        self._classify(node.args[0])
        l_zero = self.new_label()
        l_neg = self.new_label()
        l_done = self.new_label()
        self.emit(f"cmp ${_INF}, %eax")
        self.emit(f"jne {l_zero}")
        self.emit("test %r8d, %r8d")
        self.emit(f"jnz {l_neg}")
        self.emit("mov $1, %eax")
        self.emit(f"jmp {l_done}")
        self.label(l_neg)
        self.emit("mov $-1, %eax")
        self.emit(f"jmp {l_done}")
        self.label(l_zero)
        self.emit("xor %eax, %eax")
        self.label(l_done)

    def _classify(self, x) -> None:
        """Evaluate `x` and leave its category in %eax and sign in %r8d."""
        self.gen_expr(x)
        k = x.ty.kind
        if k == TypeKind.FLOAT16:
            self._classify_single(half=True)
        elif k == TypeKind.FLOAT:
            self._classify_single(half=False)
        elif k == TypeKind.DOUBLE:
            self.emit("movq %xmm0, %rdx")
            self.emit("mov %rdx, %r8")
            self.emit("shr $63, %r8")
            self.emit("mov %rdx, %rcx")
            self.emit("shr $52, %rcx")
            self.emit("and $0x7ff, %ecx")
            self.emit("movabs $0x000fffffffffffff, %rax")
            self.emit("and %rax, %rdx")
            self._fp_tail(0x7FF, x87=False)
        elif k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
            # pop st(0) into memory: 64-bit mantissa (explicit integer bit
            # included) and the 16-bit sign/exponent word
            self.emit("sub $16, %rsp")
            self.emit("fstpt (%rsp)")
            self.emit("mov (%rsp), %rdx")
            self.emit("movzwl 8(%rsp), %ecx")
            self.emit("add $16, %rsp")
            self.emit("mov %ecx, %r8d")
            self.emit("shr $15, %r8d")
            self.emit("and $0x7fff, %ecx")
            self._fp_tail(0x7FFF, x87=True)
        else:
            diag.error(x, "codegen: floating classification of this type is not supported",
                       "codegen")

    def _classify_single(self, half: bool) -> None:
        """float bits in %xmm0; for _Float16 the value is a float holding an exact
        half, so magnitudes below 2^-14 are subnormal halves (exponent forced to 0)."""
        self.emit("movd %xmm0, %edx")
        self.emit("mov %edx, %r8d")
        self.emit("shr $31, %r8d")
        self.emit("mov %edx, %ecx")
        self.emit("shr $23, %ecx")
        self.emit("and $0xff, %ecx")
        self.emit("and $0x7fffff, %edx")
        if half:
            # Below 2^-14 the float fraction is not the half fraction (2^-20 has
            # fraction 0 as a float), so replace it by "magnitude is nonzero".
            self.emit("movd %xmm0, %r9d")
            self.emit("and $0x7fffffff, %r9d")
            self.emit("test %r9d, %r9d")
            self.emit("setnz %r10b")
            self.emit("movzbl %r10b, %r10d")
            self.emit(f"cmp ${_HALF_MIN_NORMAL_BITS}, %r9d")
            self.emit("mov $0, %r11d")
            self.emit("cmovb %r11d, %ecx")
            self.emit("cmovb %r10d, %edx")
        self._fp_tail(0xFF, x87=False)

    def _fp_tail(self, emax: int, x87: bool) -> None:
        """Input: %ecx = biased exponent, %rdx = fraction (x87: full 64-bit
        mantissa with the explicit integer bit). Output: %eax = category."""
        l_notmax = self.new_label()
        l_normal = self.new_label()
        l_zero = self.new_label()
        l_inf = self.new_label()
        l_done = self.new_label()
        self.emit(f"cmp ${emax}, %ecx")
        self.emit(f"jne {l_notmax}")
        if x87:  # NaN iff the mantissa has a fraction bit below the integer bit
            self.emit("lea (%rdx,%rdx), %rax")
            self.emit("test %rax, %rax")
        else:
            self.emit("test %rdx, %rdx")
        self.emit(f"jz {l_inf}")
        self.emit(f"mov ${_NAN}, %eax")
        self.emit(f"jmp {l_done}")
        self.label(l_inf)
        self.emit(f"mov ${_INF}, %eax")
        self.emit(f"jmp {l_done}")
        self.label(l_notmax)
        self.emit("test %ecx, %ecx")
        self.emit(f"jnz {l_normal}")
        self.emit("test %rdx, %rdx")
        self.emit(f"jz {l_zero}")
        self.emit(f"mov ${_SUB}, %eax")
        self.emit(f"jmp {l_done}")
        self.label(l_zero)
        self.emit(f"mov ${_ZERO}, %eax")
        self.emit(f"jmp {l_done}")
        self.label(l_normal)
        self.emit(f"mov ${_NORMAL}, %eax")
        self.label(l_done)
