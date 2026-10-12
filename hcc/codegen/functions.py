"""Function emission: frame layout, prologue/epilogue, incoming parameters.

FunctionMixin.emit_function(fn):
  * classify params like calls.py does; stack-passed params keep their
    caller-frame slot (positive %rbp offset), register params get locals
  * assign negative %rbp offsets to all other locals
  * prologue spills argument registers (and the va_area for variadics)
  * epilogue label `.L.return.<name>`
Struct return: >16 bytes uses the hidden pointer saved at self.ret_ptr_offset.
"""
from __future__ import annotations

from hcc.codegen.abi import abi_kind, is_memory_struct, struct_arg_regs, struct_in_regs
from hcc.codegen.context import ARG_REGS8, ARG_REGS16, ARG_REGS32, ARG_REGS64
from hcc.typesys import TypeKind, align_to


class FunctionMixin:

    def _classify_params(self, fn):
        """Returns list of (var, where) with where in {'gp','fp','struct','stack'}."""
        gp = fp = 0
        stack_off = 16
        ret = fn.ty.return_ty
        if abi_kind(ret) in (TypeKind.STRUCT, TypeKind.UNION) and is_memory_struct(ret):
            gp = 1
        result = []
        for p in fn.params:
            ty = p.ty
            k = abi_kind(ty)
            where = None
            if k in (TypeKind.STRUCT, TypeKind.UNION):
                regs = struct_arg_regs(ty)
                if regs is not None:
                    fp1, fp2 = regs
                    nfp = int(fp1) + (int(fp2) if ty.size > 8 else 0)
                    ngp = (0 if fp1 else 1) + ((0 if fp2 else 1) if ty.size > 8 else 0)
                    if fp + nfp <= 8 and gp + ngp <= 6:
                        where = ("struct", gp, fp)
                        fp += nfp
                        gp += ngp
            elif k in (TypeKind.FLOAT, TypeKind.DOUBLE):
                if fp < 8:
                    where = ("fp", fp)
                fp += 1
            elif k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
                pass
            else:
                if gp < 6:
                    where = ("gp", gp)
                gp += 1
            if where is None:
                stack_off = align_to(stack_off, max(8, ty.align if ty.align <= 16 else 16))
                where = ("stack", stack_off)
                stack_off += align_to(ty.size, 8)
            result.append((p, where))
        return result, min(gp, 6), min(fp, 8), stack_off

    def layout_frame(self, fn, stack_params) -> int:
        offset = 0
        ret = fn.ty.return_ty
        self.ret_ptr_offset = None
        if abi_kind(ret) in (TypeKind.STRUCT, TypeKind.UNION) and is_memory_struct(ret):
            offset += 8
            self.ret_ptr_offset = -offset
        # Lay out in reverse declaration order so that, as with gcc -O0, the
        # first-declared local sits at the lowest address.
        for v in reversed(fn.locals):
            if v in stack_params:
                continue
            al = v.align or v.ty.align or 1
            if v.ty.kind == TypeKind.ARRAY and v.ty.size >= 16:
                al = max(al, 16)
            offset += max(v.ty.size, 0)
            offset = align_to(offset, al)
            v.offset = -offset
        return align_to(offset, 16)

    def emit_function(self, fn) -> None:
        self.fn = fn
        sym = fn.symbol
        self.directive(".text" if not fn.section else f".section {fn.section},\"ax\",@progbits")
        if not fn.is_static:
            self.directive(f".globl {sym}")
        if fn.is_weak:
            self.directive(f".weak {sym}")
        if fn.visibility in ("hidden", "internal", "protected"):
            self.directive(f".{fn.visibility} {sym}")
        self.directive(f".type {sym}, @function")
        self.label(sym)
        params, gp_named, fp_named, stack_end = self._classify_params(fn)
        stack_params = {p for p, w in params if w[0] == "stack"}
        for p, w in params:
            if w[0] == "stack":
                p.offset = w[1]
        size = self.layout_frame(fn, stack_params)
        self.va_info = (gp_named, fp_named, stack_end)
        self.emit("push %rbp")
        self.emit("mov %rsp, %rbp")
        if size:
            self.emit(f"sub ${size}, %rsp")
        if fn.chain_var is not None:
            self.emit(f"mov %r10, {fn.chain_var.offset}(%rbp)")
        self.depth = 0
        if self.ret_ptr_offset is not None:
            self.emit(f"mov %rdi, {self.ret_ptr_offset}(%rbp)")
        if fn.va_area is not None:
            off = fn.va_area.offset
            for i, r in enumerate(ARG_REGS64):
                self.emit(f"mov {r}, {off + i * 8}(%rbp)")
            for i in range(8):
                self.emit(f"movsd %xmm{i}, {off + 48 + i * 16}(%rbp)")
        for p, w in params:
            if w[0] == "gp":
                self.store_gp_param(w[1], p.offset, p.ty.size)
            elif w[0] == "fp":
                ins = "movss" if p.ty.kind == TypeKind.FLOAT else "movsd"
                self.emit(f"{ins} %xmm{w[1]}, {p.offset}(%rbp)")
            elif w[0] == "struct":
                self.store_struct_param(p, w[1], w[2])
        self.gen_stmt(fn.body)
        assert self.depth == 0, f"stack depth {self.depth} in {fn.name}"
        self.emit("mov $0, %eax")
        self.label(f".L.return.{sym}")
        self.emit("mov %rbp, %rsp")
        self.emit("pop %rbp")
        self.emit("ret")
        self.directive(f".size {sym}, .-{sym}")
        self.fn = None

    def store_gp_param(self, reg: int, off: int, size: int) -> None:
        if size == 1:
            self.emit(f"mov {ARG_REGS8[reg]}, {off}(%rbp)")
        elif size == 2:
            self.emit(f"mov {ARG_REGS16[reg]}, {off}(%rbp)")
        elif size == 4:
            self.emit(f"mov {ARG_REGS32[reg]}, {off}(%rbp)")
        else:
            self.emit(f"mov {ARG_REGS64[reg]}, {off}(%rbp)")

    def store_struct_param(self, p, gp: int, fp: int) -> None:
        ty = p.ty
        fp1, fp2 = struct_in_regs(ty)
        for chunk, is_fp in enumerate([fp1, fp2][: 1 if ty.size <= 8 else 2]):
            base = p.offset + chunk * 8
            n = min(8, ty.size - chunk * 8)
            if is_fp:
                self.emit(f"{'movss' if n == 4 else 'movsd'} %xmm{fp}, {base}(%rbp)")
                fp += 1
            else:
                reg = ARG_REGS64[gp]
                gp += 1
                # %r11 is scratch: later GP argument registers (e.g. %r8)
                # may still hold incoming arguments.
                self.emit(f"mov {reg}, %r11")
                for b in range(n):
                    self.emit(f"mov %r11b, {base + b}(%rbp)")
                    self.emit("shr $8, %r11")
