"""Variadic functions: va_start, va_arg, va_copy (SysV va_list layout).

va_list = struct { u32 gp_offset; u32 fp_offset; void *overflow_arg_area;
void *reg_save_area; }. The callee's 176-byte register save area is
fn.va_area (rdi..r9 at +0..40, xmm0..7 at +48 + 16*i), filled in the
prologue by functions.py. self.va_info = (named gp, named fp, stack end).
"""
from __future__ import annotations

from hcc import diag
from hcc.codegen.abi import struct_arg_regs
from hcc.typesys import TypeKind, align_to


class VarargsMixin:

    def builtin_va_start(self, node) -> None:
        if self.fn is None or self.fn.va_area is None:
            diag.error(node, "va_start used in function with fixed arguments", "codegen")
        gp, fp, stack_end = self.va_info
        self.gen_expr(node.args[0])
        self.emit(f"movl ${gp * 8}, (%rax)")
        self.emit(f"movl ${48 + fp * 16}, 4(%rax)")
        self.frame_lea(stack_end, "%rdx")
        self.emit("mov %rdx, 8(%rax)")
        self.frame_lea(self.fn.va_area.offset, "%rdx")
        self.emit("mov %rdx, 16(%rax)")

    def builtin_va_copy(self, node) -> None:
        self.gen_expr(node.args[1])
        self.push()
        self.gen_expr(node.args[0])
        self.pop("%rdi")
        for off in (0, 8, 16):
            self.emit(f"mov {off}(%rdi), %rdx")
            self.emit(f"mov %rdx, {off}(%rax)")

    def va_arg_split_struct(self, ty, regs, stack: str, done: str) -> None:
        """A <=16-byte struct with SSE and/or INTEGER eightbytes: its
        eightbytes live in separate save-area slots, so copy each one into
        the function's va_tmp scratch slot and leave its address in %rax.
        Expects the va_list pointer in %rcx; jumps to `stack` when the
        registers needed are exhausted (nothing is consumed then)."""
        chunks = [regs[0]] + ([regs[1]] if ty.size > 8 else [])
        nfp = sum(1 for f in chunks if f)
        ngp = len(chunks) - nfp
        if ngp:
            self.emit("movl (%rcx), %eax")
            self.emit(f"cmp ${48 - 8 * ngp}, %eax")
            self.emit(f"ja {stack}")
        if nfp:
            self.emit("movl 4(%rcx), %eax")
            self.emit(f"cmp ${176 - 16 * nfp}, %eax")
            self.emit(f"ja {stack}")
        base = self.fn.va_tmp.offset
        for i, is_fp in enumerate(chunks):
            # Each eightbyte is read as 8 bytes from its 8- or 16-byte slot.
            off_field = 4 if is_fp else 0
            self.emit(f"movl {off_field}(%rcx), %edx")
            self.emit("mov 16(%rcx), %rax")
            self.emit("add %rdx, %rax")
            self.emit("mov (%rax), %rax")
            self.emit(f"mov %rax, {base + 8 * i}(%rbp)")
            self.emit(f"addl ${16 if is_fp else 8}, {off_field}(%rcx)")
        self.frame_lea(base, "%rax")
        self.emit(f"jmp {done}")

    def builtin_va_arg(self, node) -> None:
        """Leaves the ADDRESS of the next argument in %rax, then loads it."""
        ty = node.type_arg
        self.gen_expr(node.args[0])
        self.emit("mov %rax, %rcx")
        k = ty.kind
        stack, done = self.new_label(), self.new_label()
        if k in (TypeKind.FLOAT16, TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.FLOAT128):
            # _Float128 is SSE class too: one 16-byte xmm slot (fp_offset <= 160 fits)
            self.emit("movl 4(%rcx), %eax")
            self.emit("cmp $160, %eax")
            self.emit(f"ja {stack}")
            self.emit("addl $16, 4(%rcx)")
            self.emit("add 16(%rcx), %rax")
            self.emit(f"jmp {done}")
        elif k in (TypeKind.STRUCT, TypeKind.UNION):
            regs = struct_arg_regs(ty)
            if regs is not None and not any(regs):
                need = 8 if ty.size <= 8 else 16
                self.emit("movl (%rcx), %eax")
                self.emit(f"cmp ${48 - need}, %eax")
                self.emit(f"ja {stack}")
                self.emit(f"addl ${need}, (%rcx)")
                self.emit("add 16(%rcx), %rax")
                self.emit(f"jmp {done}")
            elif regs is not None:
                self.va_arg_split_struct(ty, regs, stack, done)
        elif k == TypeKind.INT128:          # two INTEGER eightbytes in consecutive GP slots
            self.emit("movl (%rcx), %eax")
            self.emit("cmp $32, %eax")
            self.emit(f"ja {stack}")
            self.emit("addl $16, (%rcx)")
            self.emit("add 16(%rcx), %rax")
            self.emit(f"jmp {done}")
        elif k not in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
            self.emit("movl (%rcx), %eax")
            self.emit("cmp $40, %eax")
            self.emit(f"ja {stack}")
            self.emit("addl $8, (%rcx)")
            self.emit("add 16(%rcx), %rax")
            self.emit(f"jmp {done}")
        self.label(stack)
        self.emit("mov 8(%rcx), %rax")
        if ty.align > 8:
            self.emit("add $15, %rax")
            self.emit("and $-16, %rax")
        self.emit(f"lea {align_to(ty.size, 8)}(%rax), %rdx")
        self.emit("mov %rdx, 8(%rcx)")
        self.label(done)
        self.load(ty)
