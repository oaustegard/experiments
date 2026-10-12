"""Function calls (SysV x86-64 ABI, chibicc-style).

CallMixin.expr_Call(node):
  1. classify each argument: INTEGER reg / SSE reg / stack (structs > 16
     bytes, long double, or registers exhausted)
  2. pad so %rsp is 16-byte aligned at the call
  3. push stack args (right to left), then register args (right to left)
  4. evaluate callee address, pop register args, set %al = #SSE regs, call
  5. normalize small integer returns; copy small struct returns to ret_buffer
"""
from __future__ import annotations

from hcc.codegen.abi import abi_kind, is_memory_struct, struct_arg_regs, struct_in_regs
from hcc.codegen.context import ARG_REGS64
from hcc.typesys import TypeKind, align_to


def _stack_footprint(ty) -> int:
    """Bytes a stack-passed argument of type ty occupies (as pushed)."""
    if abi_kind(ty) in (TypeKind.STRUCT, TypeKind.UNION):
        return align_to(ty.size, 8)
    if ty.kind in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
        return 16
    return 8


class CallMixin:

    def _classify_args(self, node):
        """Returns (on_stack: list[bool], stack_slots: int)."""
        gp = fp = stack = 0
        ret = node.ty
        if abi_kind(ret) in (TypeKind.STRUCT, TypeKind.UNION) and is_memory_struct(ret):
            gp += 1
        on_stack = []
        for a in node.args:
            ty = a.ty
            k = abi_kind(ty)
            by_stack = False
            if k in (TypeKind.STRUCT, TypeKind.UNION):
                regs = struct_arg_regs(ty)
                if regs is None:
                    by_stack = True
                else:
                    fp1, fp2 = regs
                    nfp = int(fp1) + (int(fp2) if ty.size > 8 else 0)
                    ngp = (0 if fp1 else 1) + ((0 if fp2 else 1) if ty.size > 8 else 0)
                    if fp + nfp <= 8 and gp + ngp <= 6:
                        fp += nfp
                        gp += ngp
                    else:
                        by_stack = True
                if by_stack:
                    stack += align_to(ty.size, 8) // 8
            elif k in (TypeKind.FLOAT, TypeKind.DOUBLE):
                if fp >= 8:
                    by_stack = True
                    stack += 1
                fp += 1
            elif k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
                by_stack = True
                stack += 2
            else:
                if gp >= 6:
                    by_stack = True
                    stack += 1
                gp += 1
            on_stack.append(by_stack)
        return on_stack, stack

    def _stack_layout(self, node, on_stack):
        """Byte offsets of stack-passed args (relative to the %rsp at the
        call) and the total size; mirrors functions._classify_params."""
        offs = {}
        soff = 0
        for i, a in enumerate(node.args):
            if not on_stack[i]:
                continue
            soff = align_to(soff, max(8, min(a.ty.align, 16)))
            offs[i] = soff
            soff += _stack_footprint(a.ty)
        return offs, soff

    def _push_arg(self, a) -> None:
        self.gen_expr(a)
        k = abi_kind(a.ty)
        if k in (TypeKind.STRUCT, TypeKind.UNION):
            sz = align_to(a.ty.size, 8)
            self.emit(f"sub ${sz}, %rsp")
            self.depth += sz // 8
            self.emit("mov %rsp, %rdi")
            self.copy_struct(a.ty.size)
        elif k in (TypeKind.FLOAT, TypeKind.DOUBLE):
            self.pushf()
        elif k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
            self.emit("sub $16, %rsp")
            self.emit("fstpt (%rsp)")
            self.depth += 2
        else:
            self.push()

    def expr_Call(self, node) -> None:
        if node.func.ty.kind == TypeKind.FUNC and type(node.func).__name__ == "VarRef" and \
                node.func.var.name == "alloca":
            return self.builtin_alloca_call(node)
        on_stack, _ = self._classify_args(node)
        offs, total = self._stack_layout(node, on_stack)
        stack = align_to(total, 8) // 8
        if (self.depth + stack) % 2 == 1:
            self.emit("sub $8, %rsp")
            self.depth += 1
            stack += 1
        args = node.args
        # Stack args are pushed right to left; `cur` tracks the byte offset
        # (from the final %rsp) of the most recently pushed arg, and gaps
        # reproduce the callee's SysV slot alignment.
        cur = align_to(total, 8)
        for i in range(len(args) - 1, -1, -1):
            if on_stack[i]:
                gap = cur - offs[i] - _stack_footprint(args[i].ty)
                if gap:
                    self.emit(f"sub ${gap}, %rsp")
                    self.depth += gap // 8
                self._push_arg(args[i])
                cur = offs[i]
        if cur:
            self.emit(f"sub ${cur}, %rsp")
            self.depth += cur // 8
        for i in range(len(args) - 1, -1, -1):
            if not on_stack[i]:
                self._push_arg(args[i])
        ret = node.ty
        big_ret = abi_kind(ret) in (TypeKind.STRUCT, TypeKind.UNION) and is_memory_struct(ret)
        if big_ret:
            self.emit(f"lea {node.ret_buffer.offset}(%rbp), %rax")
            self.push()
        self.gen_expr(node.func)
        gp = fp = 0
        if big_ret:
            self.pop(ARG_REGS64[gp])
            gp += 1
        for i, a in enumerate(args):
            if on_stack[i]:
                continue
            k = abi_kind(a.ty)
            if k in (TypeKind.STRUCT, TypeKind.UNION):
                fp1, fp2 = struct_in_regs(a.ty)
                n = 1 if a.ty.size <= 8 else 2
                # struct occupies n slots on the stack, lowest first
                for j, is_fp in enumerate([fp1, fp2][:n]):
                    if is_fp:
                        self.popf(fp)
                        fp += 1
                    else:
                        self.pop(ARG_REGS64[gp])
                        gp += 1
            elif k in (TypeKind.FLOAT, TypeKind.DOUBLE):
                self.popf(fp)
                fp += 1
            else:
                self.pop(ARG_REGS64[gp])
                gp += 1
        self.emit("mov %rax, %r10")
        self.emit(f"mov ${fp}, %eax")
        self.emit("call *%r10")
        if stack:
            self.emit(f"add ${stack * 8}, %rsp")
            self.depth -= stack
        k = ret.kind
        if k == TypeKind.BOOL:
            self.emit("movzbl %al, %eax")
        elif k in (TypeKind.CHAR, TypeKind.SHORT) or (k == TypeKind.ENUM and ret.size < 4):
            if ret.size == 1:
                self.emit(f"{'movzbl' if ret.is_unsigned else 'movsbl'} %al, %eax")
            else:
                self.emit(f"{'movzwl' if ret.is_unsigned else 'movswl'} %ax, %eax")
        if abi_kind(ret) in (TypeKind.STRUCT, TypeKind.UNION) and not big_ret:
            self.copy_ret_regs_to_buffer(ret, node.ret_buffer.offset)
            self.emit(f"lea {node.ret_buffer.offset}(%rbp), %rax")

    def copy_ret_regs_to_buffer(self, ty, off: int) -> None:
        """After a call returning a small struct in rax/rdx/xmm0/xmm1."""
        regs = struct_in_regs(ty)
        if regs is None:
            return
        fp1, fp2 = regs
        gp_regs = ["%rax", "%rdx"]
        fp_n = gp_n = 0
        self.emit(f"lea {off}(%rbp), %rdi")
        for chunk, is_fp in enumerate([fp1, fp2][: 1 if ty.size <= 8 else 2]):
            base = chunk * 8
            n = min(8, ty.size - base)
            if is_fp:
                if n == 4:
                    self.emit(f"movss %xmm{fp_n}, {base}(%rdi)")
                else:
                    self.emit(f"movsd %xmm{fp_n}, {base}(%rdi)")
                fp_n += 1
            else:
                reg = gp_regs[gp_n]
                gp_n += 1
                self.emit(f"mov {reg}, %r8")
                for b in range(n):
                    self.emit(f"mov %r8b, {base + b}(%rdi)")
                    self.emit("shr $8, %r8")
