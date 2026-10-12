"""Statement codegen: blocks, if, loops, switch/case, break/continue, goto,
labels, return (incl. struct returns).

StatementMixin: one `stmt_<NodeClass>` method per statement node.
"""
from __future__ import annotations

from hcc.codegen.abi import abi_kind, is_memory_struct, is_x87_complex, struct_in_regs
from hcc.codegen.half import TO_ABI_HALF
from hcc.nodes import BuiltinCall as BuiltinCallNode, ExprStmt as ExprStmtNode
from hcc.sema.constexpr import try_eval_const
from hcc.typesys import TypeKind


def _sext(v: int, bits: int) -> int:
    """Interpret unsigned `v` (of `bits` width) as a signed value."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


class StatementMixin:

    def stmt_Block(self, node) -> None:
        # After __builtin_unreachable()/__builtin_trap() the rest of the block
        # is dead: only labelled parts stay (reachable by goto), so calls such
        # as `unreachable(); not_defined();` never reach the assembler.
        after_trap = False
        mark = len(self.cleanup_stack)
        for s in node.items:
            if after_trap:
                self.gen_dead_stmt(s)
            else:
                self.gen_stmt(s)
                after_trap = self._is_trap_stmt(s)
        self.end_cleanup_scope(mark, not after_trap)

    @staticmethod
    def _is_trap_stmt(s) -> bool:
        return (isinstance(s, ExprStmtNode) and isinstance(s.expr, BuiltinCallNode)
                and s.expr.name == "trap")

    def discard_value(self, ty) -> None:
        # A long double result lives on the x87 register stack; a discarded
        # value must be popped or the 8-slot stack overflows (-> NaN).
        if ty.kind == TypeKind.LDOUBLE:
            self.emit("fstp %st(0)")

    def stmt_ExprStmt(self, node) -> None:
        self.gen_expr(node.expr)
        self.discard_value(node.expr.ty)

    def stmt_VLAEnter(self, node) -> None:
        self.emit(f"mov %rsp, {node.saved.offset}(%rbp)")

    def stmt_VLAExit(self, node) -> None:
        self.emit(f"mov {node.saved.offset}(%rbp), %rcx")
        self._keep_alloca_memory("%rcx")
        self.emit("mov %rcx, %rsp")

    def _keep_alloca_memory(self, reg) -> None:
        """alloca memory lives until function return: never raise %rsp above
        the lowest alloca block (the function's alloca_bottom mark)."""
        mark = self.fn.alloca_bottom
        if mark is None:
            return
        self.emit(f"mov {mark.offset}(%rbp), %r10")
        self.emit(f"cmp %r10, {reg}")
        self.emit(f"cmova %r10, {reg}")

    def stmt_VLAAlloc(self, node) -> None:
        # Reset %rsp to the block's saved value first so that re-executing a
        # declaration (goto backwards) frees the previous storage.
        self.gen_expr(node.size)
        self.emit("add $15, %rax")
        self.emit("and $-16, %rax")
        self.emit(f"mov {node.saved.offset}(%rbp), %rcx")
        self._keep_alloca_memory("%rcx")
        self.emit("sub %rax, %rcx")
        self.emit("mov %rcx, %rsp")
        self.emit(f"mov %rsp, {node.var.offset}(%rbp)")

    def stmt_VLAZero(self, node) -> None:
        # Empty braces (GNU/C23): zero the whole variable-length object.
        self.gen_expr(node.size)
        self.emit("mov %rax, %rcx")
        self.emit(f"mov {node.var.offset}(%rbp), %rdi")
        self.emit("mov $0, %al")
        self.emit("rep stosb")

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
            if dead is not None:
                # Labels inside the dead branch are reachable by goto; falling
                # out of them must not run the live branch.
                self.emit(f"jmp {end}")
            self.label(skip)
            if live is not None:
                self.gen_stmt(live)
        self.label(end)

    def gen_dead_stmt(self, node) -> None:
        if node is None:
            return
        start = len(self.out)
        was_dead, self.dead_code = self.dead_code, True
        try:
            self.gen_stmt(node)
        finally:
            self.dead_code = was_dead
        seg = self.out[start:]
        first = next((i for i, ln in enumerate(seg)
                      if ln.endswith(":") and not ln.startswith(" ")), len(seg))
        keep = [ln for ln in seg[:first] if ln.startswith("  .")]
        self.out[start:] = keep + seg[first:]

    def stmt_For(self, node) -> None:
        begin = self.new_label()
        self.reject_loop_cleanup(node.init)
        self.note_loop_cleanups(node.brk_label, node.cont_label)
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
            self.discard_value(node.inc.ty)
        self.emit(f"jmp {begin}")
        self.label(node.brk_label)

    def stmt_DoWhile(self, node) -> None:
        begin = self.new_label()
        self.note_loop_cleanups(node.brk_label, node.cont_label)
        self.label(begin)
        if node.body is not None:
            self.gen_stmt(node.body)
        self.label(node.cont_label)
        self.gen_expr(node.cond)
        self.cmp_zero(node.cond.ty)
        self.emit(f"jne {begin}")
        self.label(node.brk_label)

    def stmt_Switch(self, node) -> None:
        self.note_loop_cleanups(node.brk_label)
        self.gen_expr(node.cond)
        wide = node.cond.ty.size == 8
        ax, di = ("%rax", "%rdi") if wide else ("%eax", "%edi")
        cases = node.cases
        if node.cond.ty.kind == TypeKind.INT128:
            self.int128_switch_cases(cases)     # compares both words
            cases = []
        for c in cases:
            if c.lo == c.hi:
                v = c.lo
                if -(1 << 31) <= v < (1 << 31):
                    self.emit(f"cmp ${v}, {ax}")
                else:
                    self.emit(f"movabs ${v & ((1 << 64) - 1)}, %rdi")
                    self.emit(f"cmp {di}, {ax}")
                self.emit(f"je {c.label}")
            else:
                self.case_range_test(c, wide, ax, di)
                self.emit(f"jbe {c.label}")
        if node.default is not None:
            self.emit(f"jmp {node.default.label}")
        self.emit(f"jmp {node.brk_label}")
        if node.body is not None:
            self.gen_stmt(node.body)
        self.label(node.brk_label)

    def case_range_test(self, c, wide: bool, ax: str, di: str) -> None:
        """Set flags so that `jbe` is taken iff lo <= value <= hi (unsigned).

        x86 immediates are 32-bit (sign-extended for 64-bit ops), so
        constants that do not fit go through %rcx via movabs/mov.
        """
        mask = (1 << (64 if wide else 32)) - 1
        lo = c.lo & mask
        span = (c.hi - c.lo) & mask
        self.emit(f"mov {ax}, {di}")
        if wide and not (-(1 << 31) <= _sext(lo, 64) < (1 << 31)):
            self.emit(f"movabs ${lo}, %rcx")
            self.emit(f"sub %rcx, {di}")
        else:
            self.emit(f"sub ${_sext(lo, 64) if wide else lo}, {di}")
        if span < (1 << 31):
            self.emit(f"cmp ${span}, {di}")
        else:
            self.emit(f"movabs ${span}, %rcx" if wide else f"mov ${span}, %ecx")
            self.emit(f"cmp %rcx, {di}" if wide else f"cmp %ecx, {di}")

    def stmt_Case(self, node) -> None:
        self.label(node.label)
        if node.body is not None:
            self.gen_stmt(node.body)

    def stmt_Break(self, node) -> None:
        self.cleanup_break(node.label)
        self.emit(f"jmp {node.label}")

    def stmt_Continue(self, node) -> None:
        self.cleanup_break(node.label)
        self.emit(f"jmp {node.label}")

    def stmt_Goto(self, node) -> None:
        self.reject_goto_cleanup(node)
        owner = self.fn
        while owner.outer is not None and node.name not in owner.labels:
            owner = owner.outer
        if owner is self.fn:
            self.emit(f"jmp {self.user_label(node.name)}")
            return
        # GNU non-local goto from a nested function into an enclosing one:
        # restore the enclosing %rbp and its frame-bottom %rsp (labels are
        # reached at statement depth 0, where %rsp == %rbp - frame size).
        self.frame_of(owner)
        self.emit("mov %r11, %rbp")
        self.emit(f"lea -.Lframe.{owner.symbol}(%rbp), %rsp")
        self.emit(f"jmp .L.{owner.name}.{node.name}")

    def stmt_GotoExpr(self, node) -> None:
        self.reject_goto_cleanup(node)
        self.gen_expr(node.target)
        self.emit("jmp *%rax")

    def stmt_Label(self, node) -> None:
        self.label(self.user_label(node.name))
        if node.body is not None:
            self.gen_stmt(node.body)

    def stmt_Return(self, node) -> None:
        if node.size is not None:
            self.return_dyn_struct(node)
            self.cleanup_return(node.expr.ty)
            self.emit(f"jmp .L.return.{self.fn.symbol}")
            return
        if node.expr is not None:
            self.gen_expr(node.expr)
            ty = node.expr.ty
            if ty.kind == TypeKind.FLOAT16:
                self.emit(TO_ABI_HALF)
            if is_x87_complex(ty):
                self.emit("fldt 16(%rax)")   # imaginary part first: real ends up in st(0)
                self.emit("fldt (%rax)")
            elif abi_kind(ty) in (TypeKind.STRUCT, TypeKind.UNION):
                if ty.size == 0:
                    pass  # empty struct: nothing to return
                elif not is_memory_struct(ty):
                    self.struct_to_regs(ty)
                else:
                    self.emit(f"mov {self.ret_ptr_offset}(%rbp), %rdi")
                    self.copy_struct(ty.size)
                    self.emit("mov %rdi, %rax")
        self.cleanup_return(node.expr.ty if node.expr is not None else None)
        self.emit(f"jmp .L.return.{self.fn.symbol}")

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
