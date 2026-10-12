"""Inline assembly statements.

AsmMixin.stmt_Asm(node): basic asm is emitted verbatim. Extended asm with
operands is TODO (needs constraint handling: r, m, i, a/b/c/d/S/D, =, +, &).
"""
from __future__ import annotations

from hcc import diag


class AsmMixin:

    def stmt_Asm(self, node) -> None:
        if node.outputs or node.inputs:
            if not node.text.strip():
                return
            diag.error(node, "codegen: extended asm operands are not supported yet", "codegen")
        self.out.append(node.text)
