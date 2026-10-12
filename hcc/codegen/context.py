"""CodeGenBase: shared codegen state and primitive emit helpers.

State (all on the CodeGen instance, nothing global):
    self.out     list[str] of asm lines      self.depth   pushed 8-byte slots
    self.fn      current nodes.Function      self.opts    Options
Helpers: emit(), label(), push(), pop(reg), pushf(), popf(n), new_label(),
gen_expr(node) / gen_stmt(node) dispatch to expr_<Class> / stmt_<Class>.
"""
from __future__ import annotations

from hcc import diag

ARG_REGS64 = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"]
ARG_REGS32 = ["%edi", "%esi", "%edx", "%ecx", "%r8d", "%r9d"]
ARG_REGS16 = ["%di", "%si", "%dx", "%cx", "%r8w", "%r9w"]
ARG_REGS8 = ["%dil", "%sil", "%dl", "%cl", "%r8b", "%r9b"]


from hcc.codegen.peephole import fuse_setcc_branch

class CodeGenBase:

    def __init__(self, tu, opts=None):
        self.tu = tu
        self.opts = opts
        self.out: list[str] = []
        self.depth = 0
        self.label_count = 0
        self.fn = None
        self.cleanup_stack: list = []   # pending __attribute__((cleanup)) calls (codegen/cleanup.py)
        self.cleanup_marks: dict = {}   # loop/switch label -> cleanup_stack depth at that loop
        self.dead_code = False       # set while generating a statically dead branch
        self._expr_dispatch: dict = {}
        self._stmt_dispatch: dict = {}

    # ---- output -------------------------------------------------------------
    def emit(self, line: str) -> None:
        if line[:2] in ("je", "jn") and fuse_setcc_branch(self.out, line):
            return
        self.out.append("  " + line)

    def label(self, name: str) -> None:
        self.out.append(name + ":")

    def directive(self, line: str) -> None:
        self.out.append("  " + line)

    def new_label(self) -> str:
        self.label_count += 1
        return f".L.hcc.{self.label_count}"

    def user_label(self, name: str) -> str:
        return f".L.{self.fn.name}.{name}"

    # ---- stack ----------------------------------------------------------------
    def push(self) -> None:
        self.emit("push %rax")
        self.depth += 1

    def pop(self, reg: str) -> None:
        self.emit(f"pop {reg}")
        self.depth -= 1

    def pushf(self) -> None:
        self.emit("sub $8, %rsp")
        self.emit("movsd %xmm0, (%rsp)")
        self.depth += 1

    def popf(self, n: int) -> None:
        self.emit(f"movsd (%rsp), %xmm{n}")
        self.emit("add $8, %rsp")
        self.depth -= 1

    # ---- dispatch -------------------------------------------------------------
    def gen_expr(self, node) -> None:
        cls = type(node)
        fn = self._expr_dispatch.get(cls)
        if fn is None:
            fn = getattr(self, "expr_" + cls.__name__, None)
            if fn is None:
                diag.error(node, f"codegen: unsupported expression {cls.__name__}", "codegen")
            self._expr_dispatch[cls] = fn
        fn(node)

    def gen_stmt(self, node) -> None:
        cls = type(node)
        fn = self._stmt_dispatch.get(cls)
        if fn is None:
            fn = getattr(self, "stmt_" + cls.__name__, None)
            if fn is None:
                diag.error(node, f"codegen: unsupported statement {cls.__name__}", "codegen")
            self._stmt_dispatch[cls] = fn
        self.hoist_dyn_allocs(node)
        fn(node)
