"""Inline expansion of GCC's __builtin_[rev_]crc<N>_data<D>(crc, data, poly).

All work is done in 64-bit registers. Operands are first masked to their width.
Plain CRC (MSB first): crc and poly are left-aligned (shifted by 64-N) and data is
xor-ed into the top D bits, then D steps of "shift left; if the bit that fell out was
set, xor the polynomial". The result is shifted back down.
Reflected (rev_): everything stays right-aligned, data is xor-ed into the low D bits
and each of the D steps is "shift right; if the bit that fell out was set, xor the
polynomial". The polynomial is used as given (already in reflected form).
"""
from __future__ import annotations


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
