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
            elif k in (TypeKind.FLOAT16, TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.DEC32, TypeKind.DEC64, TypeKind.DEC128):
                if fp < 8:
                    where = ("fp", fp)
                fp += 1
            elif k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
                pass
            elif k == TypeKind.INT128:
                if gp + 2 <= 6:
                    where = ("i128", gp)
                    gp += 2
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
        # Stack protector (as gcc's -fstack-protector-strong): a function with a
        # local array gets a canary just below the saved %rbp.  All array locals
        # are placed directly below the canary, so an overflow reaches the canary
        # before any scalar or the return address.
        self.canary_offset = None
        arrays = [v for v in fn.locals if v not in stack_params and v.ty.kind == TypeKind.ARRAY]
        if arrays:
            offset += 8
            self.canary_offset = -offset
            for v in reversed(arrays):
                al = v.align or v.ty.align or 1
                if v.ty.size >= 16:
                    al = max(al, 16)
                offset = align_to(offset + max(v.ty.size, 0), al)
                v.offset = -offset
        # Other locals are grouped by block (Var.scope, a path of block ids).  A block's
        # locals sit below its ancestors' locals; sibling blocks start at the same
        # base, so their slots overlap (as gcc -O0 does).  Within a group, the
        # first-declared local sits at the lowest address.
        groups: dict[tuple, list] = {}
        for v in reversed(fn.locals):
            if v in stack_params or v.ty.kind == TypeKind.ARRAY:   # arrays: placed above
                continue
            groups.setdefault(v.scope, []).append(v)
        children: dict[tuple, set] = {}
        for sc in groups:
            for k in range(len(sc)):
                children.setdefault(sc[:k], set()).add(sc[:k + 1])
        frame_end = offset
        pending = [((), offset)]
        while pending:
            sc, cur = pending.pop()
            for v in groups.get(sc, ()):
                al = v.align or v.ty.align or 1
                if v.ty.kind == TypeKind.ARRAY and v.ty.size >= 16:
                    al = max(al, 16)
                cur += max(v.ty.size, 0)
                cur = align_to(cur, al)
                v.offset = -cur
            frame_end = max(frame_end, cur)
            pending.extend((c, cur) for c in children.get(sc, ()))
        return align_to(frame_end, 16)

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
        if size and size < (1 << 31):
            self.emit(f"sub ${size}, %rsp")
        elif size:
            self.emit(f"movabs ${size}, %r11")
            self.emit("sub %r11, %rsp")
        if self.canary_offset is not None:
            self.emit("mov %fs:0x28, %r11")
            self.emit(f"mov %r11, {self.canary_offset}(%rbp)")
        if fn.alloca_bottom is not None:        # no alloca memory yet (see statements.py)
            self.emit(f"movq $-1, {fn.alloca_bottom.offset}(%rbp)")
        for var, reg in fn.reg_vars:
            self.emit(f"mov %{reg}, {var.offset}(%rbp)")
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
                if p.ty.kind == TypeKind.FLOAT16:       # ABI: half bits in low 16
                    self.emit(f"movd %xmm{w[1]}, %eax")
                    self.emit(f"mov %ax, {p.offset}(%rbp)")
                    continue
                ins = {TypeKind.FLOAT: "movss", TypeKind.DEC32: "movss",
                       TypeKind.DEC128: "movups"}.get(p.ty.kind, "movsd")
                self.emit(f"{ins} %xmm{w[1]}, {self.frame_ref(p.offset)}")
            elif w[0] == "struct":
                self.store_struct_param(p, w[1], w[2])
            elif w[0] == "i128":
                self.emit(f"mov {ARG_REGS64[w[1]]}, {p.offset}(%rbp)")
                self.emit(f"mov {ARG_REGS64[w[1] + 1]}, {p.offset + 8}(%rbp)")
        self.gen_stmt(fn.body)
        assert self.depth == 0, f"stack depth {self.depth} in {fn.name}"
        self.emit("mov $0, %eax")
        self.label(f".L.return.{sym}")
        if self.canary_offset is not None:  # %rax/%xmm0 hold the return value: use %r11 only
            ok = f".L.canary_ok.{sym}"
            self.emit(f"mov {self.canary_offset}(%rbp), %r11")
            self.emit("sub %fs:0x28, %r11")
            self.emit(f"je {ok}")
            self.emit("call __stack_chk_fail")
            self.label(ok)
        self.emit("mov %rbp, %rsp")
        self.emit("pop %rbp")
        self.emit("ret")
        self.directive(f".size {sym}, .-{sym}")
        self.directive(f".set .Lframe.{sym}, {size}")
        self.fn = None

    def frame_ref(self, off: int) -> str:
        """Memory operand for a %rbp-relative slot.  Offsets beyond disp32 (frames
        holding huge arrays) are formed in %r11, which is free in the prologue."""
        if -(1 << 31) <= off < (1 << 31):
            return f"{off}(%rbp)"
        self.emit(f"movabs ${off}, %r11")
        self.emit("add %rbp, %r11")
        return "(%r11)"

    def frame_lea(self, off, reg: str) -> None:
        """reg = address of the %rbp-relative slot at `off` (see frame_ref)."""
        if -(1 << 31) <= off < (1 << 31):
            self.emit(f"lea {off}(%rbp), {reg}")
        else:
            self.emit(f"movabs ${off}, {reg}")
            self.emit(f"add %rbp, {reg}")

    def store_gp_param(self, reg: int, off: int, size: int) -> None:
        m = self.frame_ref(off)
        if size == 1:
            self.emit(f"mov {ARG_REGS8[reg]}, {m}")
        elif size == 2:
            self.emit(f"mov {ARG_REGS16[reg]}, {m}")
        elif size == 4:
            self.emit(f"mov {ARG_REGS32[reg]}, {m}")
        else:
            self.emit(f"mov {ARG_REGS64[reg]}, {m}")

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
