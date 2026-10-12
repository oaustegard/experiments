"""__int128 codegen: 128-bit integer arithmetic, comparisons and conversions.

A 128-bit integer value lives in the accumulator pair %rdx:%rax (%rdx = high
64 bits); memory is 16 bytes little endian (load/store in codegen/memory.py).
Int128Mixin provides:
    int128_binary(node)     binary operators with 128-bit operands
    int128_unary(op)        unary - and ~ on a 128-bit operand
    int128_lit(value)       128-bit constant -> %rdx:%rax
    cast128(frm, to)        conversions where either side is __int128
Division, remainder and the float conversions call the libgcc helpers
(__divti3, __floattidf, __fixdfti, ...) that the gcc link driver provides.
"""
from __future__ import annotations

from hcc import diag
from hcc.codegen.expressions import _contains_call
from hcc.typesys import TY_LONG, TY_ULONG, TypeKind, is_flonum

_MASK64 = (1 << 64) - 1
_PTRLIKE = (TypeKind.PTR, TypeKind.ARRAY, TypeKind.FUNC)

_CMP_SIGNED = {"<": "setl", ">=": "setge", ">": "setl", "<=": "setge"}
_CMP_UNSIGNED = {"<": "setb", ">=": "setae", ">": "setb", "<=": "setae"}
_FLOAT_HELPER_SUFFIX = {TypeKind.FLOAT: "sf", TypeKind.FLOAT16: "sf", TypeKind.DOUBLE: "df",
                        TypeKind.LDOUBLE: "xf", TypeKind.FLOAT128: "xf"}


def is_i128(ty) -> bool:
    return ty is not None and ty.kind == TypeKind.INT128


def _unsigned_src(ty) -> bool:
    return ty.is_unsigned or ty.kind in _PTRLIKE


class Int128Mixin:

    # ---- stack: a 128-bit value takes two 8-byte slots -----------------------
    def push128(self) -> None:
        self.emit("push %rdx")
        self.emit("push %rax")
        self.depth += 2

    def pop128_acc(self) -> None:
        """Pop a pushed 128-bit value back into %rdx:%rax."""
        self.emit("pop %rax")
        self.emit("pop %rdx")
        self.depth -= 2

    def pop128_args(self) -> None:
        """Pop a pushed 128-bit value into %rsi:%rdi (high:low)."""
        self.emit("pop %rdi")
        self.emit("pop %rsi")
        self.depth -= 2

    def _libgcc_call(self, name: str) -> None:
        """Call a libgcc helper with the stack kept 16-byte aligned."""
        pad = self.depth % 2 == 1
        if pad:
            self.emit("sub $8, %rsp")
        self.emit(f"call {name}@PLT")
        if pad:
            self.emit("add $8, %rsp")

    # ---- literals and unary --------------------------------------------------
    def int128_lit(self, value: int) -> None:
        self.emit(f"movabs ${value & _MASK64}, %rax")
        self.emit(f"movabs ${(value >> 64) & _MASK64}, %rdx")

    def builtin_bswap128(self, node) -> None:
        """__builtin_bswap128(x): reverse the 16 bytes: bswap each 64-bit half
        and exchange the halves (%rdx:%rax in, %rdx:%rax out)."""
        self.gen_expr(node.args[0])
        self.emit("bswap %rax")
        self.emit("bswap %rdx")
        self.emit("xchg %rax, %rdx")

    def int128_unary(self, op: str, node) -> None:
        if op == "~":
            self.emit("not %rax")
            self.emit("not %rdx")
        elif op == "-":
            self.emit("neg %rax")
            self.emit("adc $0, %rdx")
            self.emit("neg %rdx")
        else:
            diag.error(node, f"codegen: unsupported operator '{op}' on __int128", "codegen")

    # ---- binary operators ----------------------------------------------------
    def int128_binary(self, node) -> None:
        if node.op not in _INT128_OPS:
            diag.error(node, f"codegen: unsupported operator '{node.op}' on __int128", "codegen")
        if _contains_call(node.lhs) and not _contains_call(node.rhs):
            # As in arith.py: a call-free operand may be parked on the stack only.
            self.gen_expr(node.lhs)
            self.push128()
            self.gen_expr(node.rhs)
            self.emit("mov %rax, %rdi")
            self.emit("mov %rdx, %rsi")
            self.pop128_acc()
        else:
            self.gen_expr(node.rhs)
            self.push128()
            self.gen_expr(node.lhs)
            self.pop128_args()           # rhs -> %rsi:%rdi, lhs stays in %rdx:%rax
        self._int128_op(node)

    def _int128_op(self, node) -> None:
        op = node.op
        uns = node.lhs.ty.is_unsigned
        e = self.emit
        if op == "+":
            e("add %rdi, %rax")
            e("adc %rsi, %rdx")
        elif op == "-":
            e("sub %rdi, %rax")
            e("sbb %rsi, %rdx")
        elif op in ("&", "|", "^"):
            ins = {"&": "and", "|": "or", "^": "xor"}[op]
            e(f"{ins} %rdi, %rax")
            e(f"{ins} %rsi, %rdx")
        elif op == "*":
            self._int128_mul()
        elif op in ("/", "%"):
            # dividend %rsi:%rdi -> (rdi, rsi); divisor (rdx, rcx) from the operands
            e("mov %rsi, %rcx")          # divisor high
            e("mov %rdi, %r8")           # divisor low
            e("mov %rdx, %rsi")          # dividend high
            e("mov %rax, %rdi")          # dividend low
            e("mov %r8, %rdx")           # divisor low
            name = ("__udivti3" if uns else "__divti3") if op == "/" else \
                ("__umodti3" if uns else "__modti3")
            self._libgcc_call(name)
        elif op in ("<<", ">>"):
            self._int128_shift(op, uns)
        elif op in ("==", "!="):
            e("xor %rdi, %rax")
            e("xor %rsi, %rdx")
            e("or %rdx, %rax")
            e(("sete" if op == "==" else "setne") + " %al")
            e("movzbl %al, %eax")
        else:
            # Ordered comparison: 128-bit subtraction flags. '>' and '<=' swap the operands.
            if op in ("<", ">="):
                e("cmp %rdi, %rax")
                e("sbb %rsi, %rdx")
            else:
                e("cmp %rax, %rdi")
                e("sbb %rdx, %rsi")
            table = _CMP_UNSIGNED if uns else _CMP_SIGNED
            e(table[op] + " %al")
            e("movzbl %al, %eax")

    def _int128_mul(self) -> None:
        """Low 128 bits of (%rdx:%rax) * (%rsi:%rdi) -> %rdx:%rax."""
        e = self.emit
        e("mov %rax, %r8")               # low1
        e("mov %rdx, %r9")               # high1
        e("imul %rdi, %r9")              # high1 * low2
        e("imul %rsi, %r8")              # low1 * high2
        e("add %r8, %r9")
        e("mul %rdi")                    # low1 * low2 -> %rdx:%rax
        e("add %r9, %rdx")

    def _int128_shift(self, op: str, uns: bool) -> None:
        """Shift %rdx:%rax by the count in %rdi (only its low 7 bits matter)."""
        e = self.emit
        e("mov %rdi, %rcx")
        if op == "<<":
            e("shld %cl, %rax, %rdx")
            e("shl %cl, %rax")
            e("xor %r8d, %r8d")
            e("test $64, %cl")
            e("cmovne %rax, %rdx")
            e("cmovne %r8, %rax")
        elif uns:
            e("shrd %cl, %rdx, %rax")
            e("shr %cl, %rdx")
            e("xor %r8d, %r8d")
            e("test $64, %cl")
            e("cmovne %rdx, %rax")
            e("cmovne %r8, %rdx")
        else:
            e("mov %rdx, %r9")
            e("sar $63, %r9")            # sign fill for counts >= 64
            e("shrd %cl, %rdx, %rax")
            e("sar %cl, %rdx")
            e("test $64, %cl")
            e("cmovne %rdx, %rax")
            e("cmovne %r9, %rdx")

    def int128_switch_cases(self, cases) -> None:
        """Dispatch jumps of a switch on a 128-bit value in %rdx:%rax."""
        for c in cases:
            if c.lo != c.hi:
                diag.error(c, "codegen: case ranges on a __int128 switch are not supported",
                           "codegen")
            skip = self.new_label()
            self.emit(f"movabs ${c.lo & _MASK64}, %rcx")
            self.emit("cmp %rcx, %rax")
            self.emit(f"jne {skip}")
            self.emit(f"movabs ${(c.lo >> 64) & _MASK64}, %rcx")
            self.emit("cmp %rcx, %rdx")
            self.emit(f"je {c.label}")
            self.label(skip)

    # ---- conversions -----------------------------------------------------------
    def cast128(self, frm, to) -> bool:
        """Conversion where `frm` or `to` is __int128. Returns False when the
        conversion is a plain narrowing to an integer type (the cast table
        handles it: the low word is already in %rax)."""
        if is_i128(to):
            if is_i128(frm):
                return True
            if is_flonum(frm):
                self._float_to_i128(frm, to)
                return True
            # integer or pointer: extend to 64 bits, then sign/zero-extend into %rdx
            if _unsigned_src(frm):
                self.cast(frm, TY_ULONG)
                self.emit("xor %edx, %edx")
            else:
                self.cast(frm, TY_LONG)
                self.emit("mov %rax, %rdx")
                self.emit("sar $63, %rdx")
            return True
        if is_flonum(to):
            self._i128_to_float(to, frm.is_unsigned)
            return True
        return False

    def _i128_to_float(self, to, unsigned: bool) -> None:
        suffix = _FLOAT_HELPER_SUFFIX.get(to.kind)
        if suffix is None:
            diag.error(to, "codegen: unsupported conversion from __int128 to this float type",
                       "codegen")
        name = "__floatuntiXX" if unsigned else "__floattiXX"
        name = name.replace("XX", suffix)
        self.emit("mov %rax, %rdi")
        self.emit("mov %rdx, %rsi")
        self._libgcc_call(name)

    def _float_to_i128(self, frm, to) -> None:
        suffix = _FLOAT_HELPER_SUFFIX.get(frm.kind)
        if suffix is None:
            diag.error(frm, "codegen: unsupported conversion to __int128 from this float type",
                       "codegen")
        name = "__fixunsXXti" if to.is_unsigned else "__fixXXti"
        name = name.replace("XX", suffix)
        if frm.kind in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
            # x87 long double arguments are passed in memory
            n = 24 if self.depth % 2 == 1 else 16
            self.emit(f"sub ${n}, %rsp")
            self.emit("fstpt (%rsp)")
            self.emit(f"call {name}@PLT")
            self.emit(f"add ${n}, %rsp")
            return
        self._libgcc_call(name)


_INT128_OPS = frozenset(["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>",
                         "==", "!=", "<", "<=", ">", ">="])

__all__ = ["Int128Mixin", "is_i128"]
