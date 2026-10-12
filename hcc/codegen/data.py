"""Static data emission: global/static variables (with relocations), .bss,
thread-locals, and top-level asm blocks.

DataMixin.emit_data(var) writes one variable; string literals are ordinary
anonymous static Vars created by the parser (see parser/primary.py).
"""
from __future__ import annotations


class DataMixin:

    def emit_data(self, var) -> None:
        sym = var.symbol
        ty = var.ty
        align = max(var.align or 1, ty.align or 1)
        if ty.kind.name == "ARRAY" and ty.size >= 16:
            align = max(align, 16)
        if not var.is_static:
            self.directive(f".globl {sym}")
        if var.is_weak:
            self.directive(f".weak {sym}")
        if var.visibility in ("hidden", "internal", "protected"):
            self.directive(f".{var.visibility} {sym}")
        size = max(ty.size, 0)
        if var.init_data is None:
            if not var.is_static and not var.is_tls and not var.section \
                    and not var.is_weak and not var.is_nocommon \
                    and sym not in self.asm_weak_syms:
                # Tentative definition (C11 6.9.2): emit as a common symbol so
                # that it merges with a real definition in another object.
                self.directive(f".comm {sym},{max(size, 1)},{align}")
                return
            if var.section:
                self.directive(f".section {var.section},\"aw\"")
            elif var.is_tls:
                self.directive(".section .tbss,\"awT\",@nobits")
            else:
                self.directive(".bss")
            self.directive(f".type {sym}, @{'tls_object' if var.is_tls else 'object'}")
            self.directive(f".size {sym}, {size}")
            self.directive(f".balign {align}")
            self.label(sym)
            self.directive(f".zero {size if size else 1}")
            return
        if var.section:
            self.directive(f".section {var.section},\"aw\"")
        elif var.is_tls:
            self.directive(".section .tdata,\"awT\",@progbits")
        else:
            self.directive(".data")
        self.directive(f".type {sym}, @{'tls_object' if var.is_tls else 'object'}")
        self.directive(f".size {sym}, {size}")
        self.directive(f".balign {align}")
        self.label(sym)
        data = var.init_data.data
        relocs = {r.offset: r for r in var.init_data.relocs}
        pos = 0
        n = len(data)
        chunk: list[str] = []
        while pos < n:
            r = relocs.get(pos)
            if r is not None:
                if chunk:
                    self.directive(".byte " + ",".join(chunk))
                    chunk = []
                sign = "+" if r.addend >= 0 else "-"
                sub = f"-{r.minus}" if r.minus else ""
                sub += "".join(f"{'+' if sg > 0 else '-'}{lab}" for sg, lab in r.extra)
                d = {1: ".byte", 2: ".short", 4: ".long"}.get(r.size, ".quad")
                self.directive(f"{d} {r.label}{sub}{sign}{abs(r.addend)}")
                pos += r.size
                continue
            chunk.append(str(data[pos]))
            if len(chunk) == 32:
                self.directive(".byte " + ",".join(chunk))
                chunk = []
            pos += 1
        if chunk:
            self.directive(".byte " + ",".join(chunk))

    def emit_toplevel_asm(self, node) -> None:
        self.out.append(node.text)
