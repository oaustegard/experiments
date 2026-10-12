"""Inline assembly statements.

AsmMixin.stmt_Asm(node): basic asm is emitted verbatim. Extended asm with
operands is TODO (needs constraint handling: r, m, i, a/b/c/d/S/D, =, +, &).
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N


class AsmMixin:

    def stmt_Asm(self, node) -> None:
        if node.outputs or node.inputs:
            if not node.text.strip():
                self._asm_operands_side_effects(node)
                return
            diag.error(node, "codegen: extended asm operands are not supported yet", "codegen")
        self.out.append(node.text)

    def _asm_operands_side_effects(self, node) -> None:
        """Empty template: no instructions, but every operand expression is
        still evaluated exactly once (lvalues by address, so no load is done)."""
        for _cons, e in list(node.outputs) + list(node.inputs):
            if type(e) in (N.VarRef, N.Deref, N.Member_):
                self.gen_addr(e)
            else:
                self.gen_expr(e)
