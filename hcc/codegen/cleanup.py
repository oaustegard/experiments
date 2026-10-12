"""__attribute__((cleanup(f))) codegen: the pending cleanup calls of a function.

CleanupMixin keeps `self.cleanup_stack`: the CleanupDecl calls (f(&var)) whose
declarations have executed, innermost last. A block runs the calls it pushed when
control falls off its end; `break`/`continue` run the calls pushed inside their
loop (the stack itself is unchanged, the code after them is still emitted);
`return` runs every pending call after the return value is computed, keeping that
value in its registers. `goto` out of a cleanup scope is rejected: not modelled.
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc.codegen.abi import is_x87_complex
from hcc.typesys import TypeKind


class CleanupMixin:

    def stmt_CleanupDecl(self, node) -> None:
        self.cleanup_stack.append(node.call)

    def run_cleanups(self, depth: int) -> None:
        """Emit the cleanup calls pushed above `depth`, innermost first."""
        for call in reversed(self.cleanup_stack[depth:]):
            self.gen_expr(call)
            self.discard_value(call.ty)

    def end_cleanup_scope(self, depth: int, falls_through: bool) -> None:
        """A block with mark `depth` ends: run its cleanups if control reaches the end,
        then forget them."""
        if falls_through:
            self.run_cleanups(depth)
        del self.cleanup_stack[depth:]

    def note_loop_cleanups(self, *labels) -> None:
        """Remember the cleanup depth at a loop/switch's break and continue labels."""
        for label in labels:
            if label:
                self.cleanup_marks[label] = len(self.cleanup_stack)

    def cleanup_break(self, label: str) -> None:
        self.run_cleanups(self.cleanup_marks.get(label, 0))

    def reject_loop_cleanup(self, init) -> None:
        """`for (T x __attribute__((cleanup(f))) = ...;;)`: the variable's scope is the
        loop, which is not modelled; refuse rather than run f at the wrong time."""
        if isinstance(init, N.Block) and any(isinstance(s, N.CleanupDecl) for s in init.items):
            diag.error(init, "cleanup attribute on a for-loop declaration is not supported",
                       "codegen")

    def reject_stmt_expr_cleanup(self, node) -> None:
        """A cleanup declared directly in `({ ... })` would run after the statement
        expression's value is in %rax (clobbering it): refuse instead."""
        if any(isinstance(s, N.CleanupDecl) for s in node.body.items):
            diag.error(node, "cleanup attribute inside a statement expression is not supported",
                       "codegen")

    def reject_goto_cleanup(self, node) -> None:
        if self.cleanup_stack:
            diag.error(node, "goto out of the scope of a cleanup variable is not supported",
                       "codegen")

    def cleanup_return(self, ty) -> None:
        """Before `return` jumps out: run every pending cleanup, preserving the return
        value (rax/rdx, xmm0/xmm1, and the x87 stack for long double results)."""
        if not self.cleanup_stack:
            return
        x87 = 0
        if ty is not None:
            if ty.kind == TypeKind.LDOUBLE:
                x87 = 1
            elif is_x87_complex(ty):
                x87 = 2
        self.emit("push %rax")
        self.emit("push %rdx")
        self.depth += 2
        self.emit("sub $16, %rsp")
        self.emit("movdqu %xmm0, (%rsp)")
        self.emit("sub $16, %rsp")
        self.emit("movdqu %xmm1, (%rsp)")
        self.depth += 4
        for _ in range(x87):        # st0 first: the lowest slot ends up holding the last one
            self.emit("sub $16, %rsp")
            self.emit("fstpt (%rsp)")
            self.depth += 2
        self.run_cleanups(0)
        for _ in range(x87):
            self.emit("fldt (%rsp)")
            self.emit("add $16, %rsp")
            self.depth -= 2
        self.emit("movdqu (%rsp), %xmm1")
        self.emit("add $16, %rsp")
        self.emit("movdqu (%rsp), %xmm0")
        self.emit("add $16, %rsp")
        self.emit("pop %rdx")
        self.emit("pop %rax")
        self.depth -= 6
