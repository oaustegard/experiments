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
from hcc.typesys import TypeKind

_NO_LOAD = (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.UNION, TypeKind.FUNC, TypeKind.VLA)


class MemoryMixin:

    def var_addr(self, var) -> None:
        if isinstance(var, N.Var) and var.is_local:
            if var.ty.kind == TypeKind.VLA:
                # VLA object: its frame slot holds the address of its storage.
                self.emit(f"mov {var.offset}(%rbp), %rax")
                return
            self.emit(f"lea {var.offset}(%rbp), %rax")
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
                self.emit(f"add ${node.member.offset}, %rax")
            return
        if t is N.Cast:
            self.gen_addr(node.operand)
            return
        if t in (N.Call, N.Assign, N.Cond, N.StmtExpr, N.BuiltinCall) and \
                node.ty.kind in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY):
            self.gen_expr(node)
            return
        diag.error(node, "lvalue required", "codegen")

    def load(self, ty) -> None:
        k = ty.kind
        if k in _NO_LOAD:
            return
        if k == TypeKind.FLOAT:
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
        elif k == TypeKind.INT128:   # __int128: %rdx:%rax (high:low)
            self.emit("mov 8(%rax), %rdx")
            self.emit("mov (%rax), %rax")
        else:
            self.emit("mov (%rax), %rax")

    def store(self, ty) -> None:
        self.pop("%rdi")
        k = ty.kind
        if k in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY):
            self.copy_struct(ty.size)
            return
        if k == TypeKind.FLOAT:
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
        elif k == TypeKind.INT128:
            self.emit("mov %rax, (%rdi)")
            self.emit("mov %rdx, 8(%rdi)")
        else:
            self.emit("mov %rax, (%rdi)")

    def copy_struct(self, size: int) -> None:
        """Copy size bytes from (%rax) to (%rdi); leaves %rax = src."""
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
