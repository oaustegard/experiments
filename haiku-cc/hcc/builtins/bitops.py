"""Bit-counting builtins: __builtin_{clz,ctz,popcount,parity,ffs,clrsb}{,l,ll}.
The suffix picks the operand width (int = 32 bit, long/long long = 64 bit); they
lower to BuiltinCall "<base>32" / "<base>64"."""
from __future__ import annotations

from hcc import typesys as T
from hcc.builtins import builtin_codegen, builtin_parser
from hcc.sema.conversions import new_cast

BIT_BUILTINS = frozenset(
    f"__builtin_{base}{suffix}"
    for base in ("clz", "ctz", "popcount", "parity", "ffs", "clrsb")
    for suffix in ("", "l", "ll"))


@builtin_parser(*sorted(BIT_BUILTINS))
def _parse_bitop(self, tok):
    """clz/ctz/popcount/parity/ffs: argument converted to unsigned int
    (or unsigned long for the l/ll forms); result is int."""
    name = tok.text[len("__builtin_"):]
    base = name.rstrip("l")
    bits = 64 if name[len(base):] else 32
    ty = T.TY_ULONG if bits == 64 else T.TY_UINT
    if base in ("ffs", "clrsb"):  # signed operands
        ty = T.TY_LONG if bits == 64 else T.TY_INT
    args = [new_cast(a, ty) for a in self._args(1)]
    return self._call(tok, f"{base}{bits}", args, T.TY_INT)


# Results are fully defined even for a zero argument (GCC leaves clz/ctz of 0
# undefined): clz/ctz give the width, ffs gives 0, popcount/parity give 0.
@builtin_codegen("clz32")
def _gen_clz32(self, node) -> None:
    _bit_arg32(self, node)
    self.emit("mov $-1, %edx")
    self.emit("bsr %eax, %eax")
    self.emit("cmovz %edx, %eax")
    self.emit("neg %eax")
    self.emit("add $31, %eax")


@builtin_codegen("clz64")
def _gen_clz64(self, node) -> None:
    self.gen_expr(node.args[0])
    self.emit("mov $-1, %edx")
    self.emit("bsr %rax, %rax")
    self.emit("cmovz %edx, %eax")
    self.emit("neg %eax")
    self.emit("add $63, %eax")


@builtin_codegen("ctz32")
def _gen_ctz32(self, node) -> None:
    _bit_arg32(self, node)
    self.emit("mov $32, %edx")
    self.emit("bsf %eax, %eax")
    self.emit("cmovz %edx, %eax")


@builtin_codegen("ctz64")
def _gen_ctz64(self, node) -> None:
    self.gen_expr(node.args[0])
    self.emit("mov $64, %edx")
    self.emit("bsf %rax, %rax")
    self.emit("cmovz %edx, %eax")


@builtin_codegen("ffs32")
def _gen_ffs32(self, node) -> None:
    _bit_arg32(self, node)
    self.emit("mov $-1, %edx")
    self.emit("bsf %eax, %eax")
    self.emit("cmovz %edx, %eax")
    self.emit("add $1, %eax")


@builtin_codegen("ffs64")
def _gen_ffs64(self, node) -> None:
    self.gen_expr(node.args[0])
    self.emit("mov $-1, %edx")
    self.emit("bsf %rax, %rax")
    self.emit("cmovz %edx, %eax")
    self.emit("add $1, %eax")


# clrsb(x): number of redundant sign bits below the sign bit, i.e.
# clz(x ^ (x >> (w-1))) - 1. Defined for 0 and -1 (gives w-1).
@builtin_codegen("clrsb32")
def _gen_clrsb32(self, node) -> None:
    self.gen_expr(node.args[0])
    self.emit("mov %eax, %edx")
    self.emit("sar $31, %edx")
    self.emit("xor %edx, %eax")
    self.emit("mov $-1, %edx")
    self.emit("bsr %eax, %eax")
    self.emit("cmovz %edx, %eax")
    self.emit("neg %eax")
    self.emit("add $30, %eax")


@builtin_codegen("clrsb64")
def _gen_clrsb64(self, node) -> None:
    self.gen_expr(node.args[0])
    self.emit("mov %rax, %rdx")
    self.emit("sar $63, %rdx")
    self.emit("xor %rdx, %rax")
    self.emit("mov $-1, %edx")
    self.emit("bsr %rax, %rax")
    self.emit("cmovz %edx, %eax")
    self.emit("neg %eax")
    self.emit("add $62, %eax")


@builtin_codegen("popcount32")
def _gen_popcount32(self, node) -> None:
    _bit_arg32(self, node)
    _popcount32(self)


@builtin_codegen("popcount64")
def _gen_popcount64(self, node) -> None:
    self.gen_expr(node.args[0])
    _popcount64(self)


@builtin_codegen("parity32")
def _gen_parity32(self, node) -> None:
    _bit_arg32(self, node)
    _popcount32(self)
    self.emit("and $1, %eax")


@builtin_codegen("parity64")
def _gen_parity64(self, node) -> None:
    self.gen_expr(node.args[0])
    _popcount64(self)
    self.emit("and $1, %eax")


def _bit_arg32(self, node) -> None:
    self.gen_expr(node.args[0])
    self.emit("mov %eax, %eax")  # clear any upper garbage


def _popcount32(self) -> None:
    """SWAR population count of %eax (clobbers %edx)."""
    self.emit("mov %eax, %edx")
    self.emit("shr $1, %edx")
    self.emit("and $0x55555555, %edx")
    self.emit("sub %edx, %eax")
    self.emit("mov %eax, %edx")
    self.emit("and $0x33333333, %eax")
    self.emit("shr $2, %edx")
    self.emit("and $0x33333333, %edx")
    self.emit("add %edx, %eax")
    self.emit("mov %eax, %edx")
    self.emit("shr $4, %edx")
    self.emit("add %edx, %eax")
    self.emit("and $0x0f0f0f0f, %eax")
    self.emit("imul $0x01010101, %eax, %eax")
    self.emit("shr $24, %eax")


def _popcount64(self) -> None:
    """SWAR population count of %rax (clobbers %rdx, %rcx)."""
    self.emit("mov %rax, %rdx")
    self.emit("shr $1, %rdx")
    self.emit("movabs $0x5555555555555555, %rcx")
    self.emit("and %rcx, %rdx")
    self.emit("sub %rdx, %rax")
    self.emit("movabs $0x3333333333333333, %rcx")
    self.emit("mov %rax, %rdx")
    self.emit("and %rcx, %rax")
    self.emit("shr $2, %rdx")
    self.emit("and %rcx, %rdx")
    self.emit("add %rdx, %rax")
    self.emit("mov %rax, %rdx")
    self.emit("shr $4, %rdx")
    self.emit("add %rdx, %rax")
    self.emit("movabs $0x0f0f0f0f0f0f0f0f, %rcx")
    self.emit("and %rcx, %rax")
    self.emit("movabs $0x0101010101010101, %rcx")
    self.emit("imul %rcx, %rax")
    self.emit("shr $56, %rax")
