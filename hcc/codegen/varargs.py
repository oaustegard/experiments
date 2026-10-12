"""Variadic functions: va_start, va_arg, va_copy (SysV va_list layout).

va_list = struct { u32 gp_offset; u32 fp_offset; void *overflow_arg_area;
void *reg_save_area; }. The callee's 176-byte register save area is
fn.va_area (rdi..r9 at +0..40, xmm0..7 at +48 + 16*i), filled in the
prologue by functions.py. self.va_info = (named gp, named fp, stack end).
"""
from __future__ import annotations

from hcc import diag
from hcc.codegen.abi import struct_in_regs
from hcc.typesys import TypeKind, align_to


class VarargsMixin:

    def builtin_va_start(self, node) -> None:
        if self.fn is None or self.fn.va_area is None:
            diag.error(node, "va_start used in function with fixed arguments", "codegen")
        gp, fp, stack_end = self.va_info
        self.gen_expr(node.args[0])
        self.emit(f"movl ${gp * 8}, (%rax)")
        self.emit(f"movl ${48 + fp * 16}, 4(%rax)")
        self.emit(f"lea {stack_end}(%rbp), %rdx")
        self.emit("mov %rdx, 8(%rax)")
        self.emit(f"lea {self.fn.va_area.offset}(%rbp), %rdx")
        self.emit("mov %rdx, 16(%rax)")

    def builtin_va_copy(self, node) -> None:
        self.gen_expr(node.args[1])
        self.push()
        self.gen_expr(node.args[0])
        self.pop("%rdi")
        for off in (0, 8, 16):
            self.emit(f"mov {off}(%rdi), %rdx")
            self.emit(f"mov %rdx, {off}(%rax)")

    def builtin_va_arg(self, node) -> None:
        """Leaves the ADDRESS of the next argument in %rax, then loads it."""
        ty = node.type_arg
        self.gen_expr(node.args[0])
        self.emit("mov %rax, %rcx")
        k = ty.kind
        stack, done = self.new_label(), self.new_label()
        if k in (TypeKind.FLOAT, TypeKind.DOUBLE):
            self.emit("movl 4(%rcx), %eax")
            self.emit("cmp $160, %eax")
            self.emit(f"ja {stack}")
            self.emit("addl $16, 4(%rcx)")
            self.emit("add 16(%rcx), %rax")
            self.emit(f"jmp {done}")
        elif k in (TypeKind.STRUCT, TypeKind.UNION):
            regs = struct_in_regs(ty)
            if regs is not None and not regs[0] and not (ty.size > 8 and regs[1]):
                need = 8 if ty.size <= 8 else 16
                self.emit("movl (%rcx), %eax")
                self.emit(f"cmp ${48 - need}, %eax")
                self.emit(f"ja {stack}")
                self.emit(f"addl ${need}, (%rcx)")
                self.emit("add 16(%rcx), %rax")
                self.emit(f"jmp {done}")
            # TODO: SSE-class and mixed structs in registers
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
