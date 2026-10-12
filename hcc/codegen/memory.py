"""Memory access: load/store by type, lvalue addresses, struct copies.

MemoryMixin provides:
    gen_addr(node)     address of an lvalue (or struct-valued expr) -> %rax
    load(ty)           %rax holds an address -> value in %rax/%xmm0/st(0)
    store(ty)          value in accumulator, address on the stack -> memory
    copy_struct(size)  copy `size` bytes from (%rax) to (%rdi)
    var_addr(var)      emit address of a Var/Function into %rax
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc.codegen.scalar_rev import load_rev_scalar, store_rev_scalar
from hcc.typesys import TypeKind
from hcc.codegen.half import LOAD_HALF, STORE_HALF

_NO_LOAD = (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.UNION, TypeKind.FUNC, TypeKind.VLA)


class MemoryMixin:

    def var_addr(self, var) -> None:
        if isinstance(var, N.Var) and var.is_local:
            if self.is_outer_local(var):
                return self.outer_var_addr(var)
            if var.ty.kind == TypeKind.VLA:
                # VLA object: its frame slot holds the address of its storage.
                self.emit(f"mov {var.offset}(%rbp), %rax")
                return
            self.lea_rbp(var.offset, "%rax")
            return
        if isinstance(var, N.Var) and var.is_tls:
            self.emit("mov %fs:0, %rax")
            self.emit(f"add ${var.symbol}@tpoff, %rax")
            return
        if self.opts is not None and getattr(self.opts, "pic", False) and not var.is_static:
            self.emit(f"mov {var.symbol}@GOTPCREL(%rip), %rax")
            return
        self.emit(f"lea {var.symbol}(%rip), %rax")

    def gen_addr(self, node) -> None:
        t = type(node)
        if t is N.VarRef:
            self.var_addr(node.var)
            return
        if t is N.Deref:
            self.gen_expr(node.operand)
            return
        if t is N.Comma:
            self.gen_expr(node.lhs)
            self.gen_addr(node.rhs)
            return
        if t is N.Member_:
            self.gen_addr(node.operand)
            if node.member.offset:
                self.add_offset_rax(node.member.offset)
            return
        if t is N.Cast:
            self.gen_addr(node.operand)
            return
        if t in (N.Call, N.Assign, N.Cond, N.StmtExpr, N.BuiltinCall) and \
                node.ty.kind in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY):
            self.gen_expr(node)
            return
        diag.error(node, "lvalue required", "codegen")

    def lea_rbp(self, off: int, reg: str) -> None:
        """reg = %rbp + off; frame slots beyond disp32 (objects > 2 GiB) are
        computed with movabs, since lea cannot take a 64-bit displacement."""
        if -0x80000000 <= off <= 0x7fffffff:
            self.emit(f"lea {off}(%rbp), {reg}")
        else:
            self.emit(f"movabs ${off}, {reg}")
            self.emit(f"add %rbp, {reg}")

    def add_offset_rax(self, off: int) -> None:
        """%rax += off. Offsets outside imm32 (huge arrays in packed structs)
        go through %r11, since x86 has no 64-bit immediate add."""
        if -0x80000000 <= off <= 0x7fffffff:
            self.emit(f"add ${off}, %rax")
        else:
            self.emit(f"movabs ${off}, %r11")
            self.emit("add %r11, %rax")

    def load(self, ty) -> None:
        k = ty.kind
        if k in _NO_LOAD:
            return
        if ty.is_rev_order:
            load_rev_scalar(self, ty)
        elif k == TypeKind.FLOAT16:
            self.emit(LOAD_HALF)
        elif k == TypeKind.FLOAT:
            self.emit("movss (%rax), %xmm0")
        elif k == TypeKind.DOUBLE:
            self.emit("movsd (%rax), %xmm0")
        elif k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
            self.emit("fldt (%rax)")
        elif ty.size == 1:
            self.emit(f"{'movzbl' if ty.is_unsigned else 'movsbl'} (%rax), %eax")
        elif ty.size == 2:
            self.emit(f"{'movzwl' if ty.is_unsigned else 'movswl'} (%rax), %eax")
        elif ty.size == 4:
            self.emit("movslq (%rax), %rax")
        elif ty.size == 16:          # __int128: low half only (TODO)
            self.emit("mov (%rax), %rax")
        else:
            self.emit("mov (%rax), %rax")

    def store(self, ty) -> None:
        self.pop("%rdi")
        k = ty.kind
        if k in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY):
            self.copy_struct(ty.size)
            return
        if ty.is_rev_order:
            store_rev_scalar(self, ty)
        elif k == TypeKind.FLOAT16:
            self.emit(STORE_HALF)
        elif k == TypeKind.FLOAT:
            self.emit("movss %xmm0, (%rdi)")
        elif k == TypeKind.DOUBLE:
            self.emit("movsd %xmm0, (%rdi)")
        elif k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
            self.emit("fstpt (%rdi)")
            self.emit("fldt (%rdi)")
        elif ty.size == 1:
            self.emit("mov %al, (%rdi)")
        elif ty.size == 2:
            self.emit("mov %ax, (%rdi)")
        elif ty.size == 4:
            self.emit("mov %eax, (%rdi)")
        elif ty.size == 16:
            self.emit("mov %rax, (%rdi)")
            self.emit("sar $63, %rax")
            self.emit("mov %rax, 8(%rdi)")
        else:
            self.emit("mov %rax, (%rdi)")

    def copy_struct(self, size: int) -> None:
        """Copy size bytes from (%rax) to (%rdi); leaves %rax = src."""
        if size > 64:
            # Large aggregates: unrolled moves would emit O(size) code, so use
            # rep movsb. rsi/rcx/rdi are saved so the caller's view is unchanged.
            self.emit("push %rsi")
            self.emit("push %rcx")
            self.emit("push %rdi")
            self.emit("mov %rax, %rsi")
            self.emit(f"mov ${size}, %rcx")
            self.emit("rep movsb")
            self.emit("pop %rdi")
            self.emit("pop %rcx")
            self.emit("pop %rsi")
            return
        i = 0
        while i + 8 <= size:
            self.emit(f"mov {i}(%rax), %r8")
            self.emit(f"mov %r8, {i}(%rdi)")
            i += 8
        while i + 4 <= size:
            self.emit(f"mov {i}(%rax), %r8d")
            self.emit(f"mov %r8d, {i}(%rdi)")
            i += 4
        while i < size:
            self.emit(f"mov {i}(%rax), %r8b")
            self.emit(f"mov %r8b, {i}(%rdi)")
            i += 1
