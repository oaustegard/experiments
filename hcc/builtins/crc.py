"""GCC CRC builtins: __builtin_[rev_]crc{8,16,32,64}_data{8,16,32,64}(crc, data, poly).

Parsing: the result and `crc`/`poly` have the width of the CRC (uint8/16/32_t); `data`
has the width of the data suffix. Lowered to a BuiltinCall named "[rev_]crc<N>_data<D>".
Being special builtins, these names never reach the libc-name lookup in primary.py
(a user function may share the name).

Codegen (inline expansion):
All work is done in 64-bit registers. Operands are first masked to their width.
Plain CRC (MSB first): crc and poly are left-aligned (shifted by 64-N) and data is
xor-ed into the top D bits, then D steps of "shift left; if the bit that fell out was
set, xor the polynomial". The result is shifted back down.
Reflected (rev_): everything stays right-aligned, data is xor-ed into the low D bits
and each of the D steps is "shift right; if the bit that fell out was set, xor the
polynomial". The polynomial is used as given (already in reflected form).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.builtins import builtin_codegen, builtin_parser
from hcc.sema.conversions import new_cast

_WIDTH_TY = {8: T.TY_UCHAR, 16: T.TY_USHORT, 32: T.TY_UINT, 64: T.TY_ULONG}

CRC_BUILTINS = frozenset(
    f"__builtin_{prefix}crc{n}_data{d}"
    for prefix in ("", "rev_")
    for n in (8, 16, 32, 64)
    for d in (8, 16, 32, 64) if d <= n
)


@builtin_parser(*sorted(CRC_BUILTINS))
def crc_call(parser, tok):
    """Parse the argument list (after '(') of a CRC builtin and return the BuiltinCall."""
    name = tok.text[len("__builtin_"):]
    rest = name.split("crc", 1)[1]               # "<N>_data<D>"
    n_bits, d_bits = (int(x) for x in rest.split("_data"))
    crc, data, poly = parser._args(3)
    args = [new_cast(crc, _WIDTH_TY[n_bits]), new_cast(data, _WIDTH_TY[d_bits]),
            new_cast(poly, _WIDTH_TY[n_bits])]
    return N.BuiltinCall(tok.loc, name, args, None, ty=_WIDTH_TY[n_bits])


def _mask_reg(gen, reg: str, bits: int) -> None:
    """Zero the bits above `bits` of the 64-bit register `reg` (%rax/%rcx/%rdx)."""
    low = {"%rax": ("%eax", "%ax", "%al"), "%rcx": ("%ecx", "%cx", "%cl"),
           "%rdx": ("%edx", "%dx", "%dl")}[reg]
    if bits == 8:
        gen.emit(f"movzbl {low[2]}, {low[0]}")
    elif bits == 16:
        gen.emit(f"movzwl {low[1]}, {low[0]}")
    elif bits == 32:
        gen.emit(f"movl {low[0]}, {low[0]}")


@builtin_codegen(*sorted(n[len("__builtin_"):] for n in CRC_BUILTINS))
def crc_builtin(gen, node) -> None:
    """Expand the CRC builtin `node` (name "[rev_]crc<N>_data<D>") into %rax."""
    rev = node.name.startswith("rev_")
    n_bits, d_bits = (int(x) for x in node.name.split("crc", 1)[1].split("_data"))
    gen.gen_expr(node.args[2])                  # poly
    gen.push()
    gen.gen_expr(node.args[1])                  # data
    gen.push()
    gen.gen_expr(node.args[0])                  # crc -> %rax
    gen.pop("%rcx")
    gen.pop("%rdx")
    _mask_reg(gen, "%rax", n_bits)
    _mask_reg(gen, "%rcx", d_bits)
    _mask_reg(gen, "%rdx", n_bits)
    label = gen.new_label()
    if rev:
        gen.emit("xor %rcx, %rax")
        for _ in range(d_bits):
            gen.emit("shr $1, %rax")
            gen.emit(f"jnc {label}")
            gen.emit("xor %rdx, %rax")
            gen.label(label)
            label = gen.new_label()
        return
    gen.emit(f"shl ${64 - n_bits}, %rax")
    gen.emit(f"shl ${64 - d_bits}, %rcx")
    gen.emit(f"shl ${64 - n_bits}, %rdx")
    gen.emit("xor %rcx, %rax")
    for _ in range(d_bits):
        gen.emit("shl $1, %rax")
        gen.emit(f"jnc {label}")
        gen.emit("xor %rdx, %rax")
        gen.label(label)
        label = gen.new_label()
    gen.emit(f"shr ${64 - n_bits}, %rax")
