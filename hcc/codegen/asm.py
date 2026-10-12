"""Inline assembly statements.

AsmMixin.stmt_Asm(node): basic asm is emitted verbatim; extended asm with
operands is lowered by codegen/asm_ext.py (constraints, %N substitution,
input/output register shuffling).
"""
from __future__ import annotations

from hcc.codegen import asm_ext


class AsmMixin:

    def stmt_Asm(self, node) -> None:
        if node.extended or node.outputs or node.inputs:
            asm_ext.lower_extended(self, node)
            return
        self.out.append(node.text)
