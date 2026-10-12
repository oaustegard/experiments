"""__builtin_bswap16/32/64/128: byte reversal (16/32/64 folded in constant expressions;
128 is lowered by Int128Mixin.builtin_bswap128 in codegen/int128.py)."""
from __future__ import annotations

from hcc import typesys as T
from hcc.builtins import builtin_codegen, builtin_fold, builtin_parser
from hcc.sema.conversions import new_cast

_TYPES = {16: T.TY_USHORT, 32: T.TY_UINT, 64: T.TY_ULONG, 128: T.TY_UINT128}


@builtin_parser("__builtin_bswap16", "__builtin_bswap32", "__builtin_bswap64", "__builtin_bswap128")
def _parse_bswap(self, tok):
    bits = int(tok.text[len("__builtin_bswap"):])
    ty = _TYPES[bits]
    args = [new_cast(a, ty) for a in self._args(1)]
    return self._call(tok, f"bswap{bits}", args, ty)


@builtin_fold("bswap16", "bswap32", "bswap64")
def _fold_bswap(node, num):
    nbytes = int(node.name[len("bswap"):]) // 8
    v = num(node.args[0]) & ((1 << (8 * nbytes)) - 1)
    return int.from_bytes(v.to_bytes(nbytes, "little"), "big")


@builtin_codegen("bswap16")
def _gen_bswap16(self, node) -> None:
    self.gen_expr(node.args[0])
    self.emit("rol $8, %ax")
    self.emit("movzwl %ax, %eax")


@builtin_codegen("bswap32")
def _gen_bswap32(self, node) -> None:
    self.gen_expr(node.args[0])
    self.emit("bswap %eax")


@builtin_codegen("bswap64")
def _gen_bswap64(self, node) -> None:
    self.gen_expr(node.args[0])
    self.emit("bswap %rax")
