"""Operand check for `asm` output operands: the operand must be a modifiable lvalue (C11 6.8.2 / GCC)."""

from __future__ import annotations

from hcc.parser.incdec_check import _is_const_lvalue


def check_asm_output_writable(parser, node, tok) -> None:
    """Reject a const-qualified object (`const int i; asm("" : "=m"(i))`) used as an output operand."""
    if _is_const_lvalue(parser.typed(node)):
        parser.error("read-only variable used as 'asm' output", tok)
