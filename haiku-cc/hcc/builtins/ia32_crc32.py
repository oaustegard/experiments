"""SSE4.2 __builtin_ia32_crc32{qi,hi,si,di}(crc, v): CRC-32C accumulated over v's width."""
from __future__ import annotations

from hcc import typesys as T
from hcc.builtins import builtin_codegen, builtin_parser
from hcc.sema.conversions import new_cast

# The value operand width gives the codegen suffix; `di` works on 64-bit operands.
CRC32_BUILTINS = {"qi": T.TY_UCHAR, "hi": T.TY_USHORT, "si": T.TY_UINT, "di": T.TY_ULLONG}


@builtin_parser(*(f"__builtin_ia32_crc32{w}" for w in CRC32_BUILTINS))
def _parse_crc32(self, tok):
    """__builtin_ia32_crc32{qi,hi,si,di}(crc, v): crc is unsigned int (unsigned
    long long for di); the result has the same type as crc."""
    width = tok.text[-2:]
    crc_ty = T.TY_ULLONG if width == "di" else T.TY_UINT
    crc, val = self._args(2)
    args = [new_cast(crc, crc_ty), new_cast(val, CRC32_BUILTINS[width])]
    return self._call(tok, f"ia32_crc32{width}", args, crc_ty)


def _crc32(self, node, insn: str, src: str, acc: str) -> None:
    """crc = CRC-32C step over the value (args[1]): value is evaluated first and
    parked on the stack so the crc expression can use %rax freely."""
    self.gen_expr(node.args[1])
    self.push()
    self.gen_expr(node.args[0])
    self.pop("%rcx")
    self.emit(f"{insn} {src}, {acc}")


@builtin_codegen("ia32_crc32qi")
def _gen_crc32qi(self, node) -> None:
    _crc32(self, node, "crc32b", "%cl", "%eax")


@builtin_codegen("ia32_crc32hi")
def _gen_crc32hi(self, node) -> None:
    _crc32(self, node, "crc32w", "%cx", "%eax")


@builtin_codegen("ia32_crc32si")
def _gen_crc32si(self, node) -> None:
    _crc32(self, node, "crc32l", "%ecx", "%eax")


@builtin_codegen("ia32_crc32di")
def _gen_crc32di(self, node) -> None:
    _crc32(self, node, "crc32q", "%rcx", "%rax")
