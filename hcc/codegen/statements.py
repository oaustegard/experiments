"""Statement codegen: blocks, if, loops, switch/case, break/continue, goto,
labels, return (incl. struct returns).

StatementMixin: one `stmt_<NodeClass>` method per statement node.
"""
from __future__ import annotations

from hcc.codegen.abi import struct_in_regs
from hcc.sema.constexpr import try_eval_const
from hcc.typesys import TypeKind


class StatementMixin:

    def stmt_Block(self, node) -> None:
        for s in node.items:
            self.gen_stmt(s)

    def stmt_ExprStmt(self, node) -> None:
        self.gen_expr(node.expr)

    def stmt_If(self, node) -> None:
        taken = try_eval_const(node.cond)
        if taken is not None:
            self.gen_const_if(node, bool(taken))
            return
        els, end = self.new_label(), self.new_label()
        self.gen_expr(node.cond)
        self.cmp_zero(node.cond.ty)
        self.emit(f"je {els}")
        self.gen_stmt(node.then)
        self.emit(f"jmp {end}")
        self.label(els)
        if node.els is not None:
            self.gen_stmt(node.els)
        self.label(end)

    def gen_const_if(self, node, taken: bool) -> None:
        """`if` with a constant condition: only the taken branch runs.

        The dead branch is still generated so that labels inside it (e.g.
        `case` labels) exist, but the instructions before its first label are
        unreachable and dropped, so calls like `link_error()` never reach asm.
        """
        end = self.new_label()
        live, dead = (node.then, node.els) if taken else (node.els, node.then)
        if taken:
            if live is not None:
                self.gen_stmt(live)
            self.emit(f"jmp {end}")
            self.gen_dead_stmt(dead)
        else:
            skip = self.new_label()
            self.emit(f"jmp {skip}")
            self.gen_dead_stmt(dead)
            self.label(skip)
            if live is not None:
                self.gen_stmt(live)
        self.label(end)

    def gen_dead_stmt(self, node) -> None:
        if node is None:
            return
        start = len(self.out)
        self.gen_stmt(node)
        seg = self.out[start:]
        first = next((i for i, ln in enumerate(seg)
                      if ln.endswith(":") and not ln.startswith(" ")), len(seg))
        keep = [ln for ln in seg[:first] if ln.startswith("  .")]
        self.out[start:] = keep + seg[first:]

    def stmt_For(self, node) -> None:
        begin = self.new_label()
        if node.init is not None:
            self.gen_stmt(node.init)
        self.label(begin)
        if node.cond is not None:
            self.gen_expr(node.cond)
            self.cmp_zero(node.cond.ty)
            self.emit(f"je {node.brk_label}")
        if node.body is not None:
            self.gen_stmt(node.body)
        self.label(node.cont_label)
        if node.inc is not None:
            self.gen_expr(node.inc)
        self.emit(f"jmp {begin}")
        self.label(node.brk_label)

    def stmt_DoWhile(self, node) -> None:
        begin = self.new_label()
        self.label(begin)
        if node.body is not None:
            self.gen_stmt(node.body)
        self.label(node.cont_label)
        self.gen_expr(node.cond)
        self.cmp_zero(node.cond.ty)
        self.emit(f"jne {begin}")
        self.label(node.brk_label)

    def stmt_Switch(self, node) -> None:
        self.gen_expr(node.cond)
        wide = node.cond.ty.size == 8
        ax, di = ("%rax", "%rdi") if wide else ("%eax", "%edi")
        for c in node.cases:
            if c.lo == c.hi:
                v = c.lo
                if -(1 << 31) <= v < (1 << 31):
                    self.emit(f"cmp ${v}, {ax}")
                else:
                    self.emit(f"movabs ${v & ((1 << 64) - 1)}, %rdi")
                    self.emit(f"cmp {di}, {ax}")
                self.emit(f"je {c.label}")
            else:
                self.emit(f"mov {ax}, {di}")
                self.emit(f"sub ${c.lo}, {di}")
                self.emit(f"cmp ${c.hi - c.lo}, {di}")
                self.emit(f"jbe {c.label}")
        if node.default is not None:
            self.emit(f"jmp {node.default.label}")
        self.emit(f"jmp {node.brk_label}")
        if node.body is not None:
            self.gen_stmt(node.body)
        self.label(node.brk_label)

    def stmt_Case(self, node) -> None:
        self.label(node.label)
        if node.body is not None:
            self.gen_stmt(node.body)

    def stmt_Break(self, node) -> None:
        self.emit(f"jmp {node.label}")

    def stmt_Continue(self, node) -> None:
        self.emit(f"jmp {node.label}")

    def stmt_Goto(self, node) -> None:
        self.emit(f"jmp {self.user_label(node.name)}")

    def stmt_GotoExpr(self, node) -> None:
        self.gen_expr(node.target)
        self.emit("jmp *%rax")

    def stmt_Label(self, node) -> None:
        self.label(self.user_label(node.name))
        if node.body is not None:
            self.gen_stmt(node.body)

    def stmt_Return(self, node) -> None:
        if node.expr is not None:
            self.gen_expr(node.expr)
            ty = node.expr.ty
            if ty.kind in (TypeKind.STRUCT, TypeKind.UNION):
                if ty.size == 0:
                    pass  # empty struct: nothing to return
                elif ty.size <= 16:
                    self.struct_to_regs(ty)
                else:
                    self.emit(f"mov {self.ret_ptr_offset}(%rbp), %rdi")
                    self.copy_struct(ty.size)
                    self.emit("mov %rdi, %rax")
        self.emit(f"jmp .L.return.{self.fn.name}")

    def struct_to_regs(self, ty) -> None:
        """%rax points at a <=16-byte struct; load it into rax/rdx/xmm0/xmm1."""
        fp1, fp2 = struct_in_regs(ty)
        self.emit("mov %rax, %rsi")
        gp_regs = ["%rax", "%rdx"]
        gp = fp = 0
        for chunk, is_fp in enumerate([fp1, fp2][: 1 if ty.size <= 8 else 2]):
            base = chunk * 8
            n = min(8, ty.size - base)
            if is_fp:
                self.emit(f"{'movss' if n == 4 else 'movsd'} {base}(%rsi), %xmm{fp}")
                fp += 1
            else:
                reg = gp_regs[gp]
                gp += 1
                if n == 8:
                    self.emit(f"mov {base}(%rsi), {reg}")
                else:
                    self.emit("xor %r8, %r8")
                    for b in range(n - 1, -1, -1):
                        self.emit("shl $8, %r8")
                        self.emit(f"mov {base + b}(%rsi), %r8b")
                    self.emit(f"mov %r8, {reg}")
