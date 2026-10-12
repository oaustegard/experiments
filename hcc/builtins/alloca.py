"""__builtin_alloca, __builtin_alloca_with_align and plain alloca(n) calls."""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.builtins import builtin_codegen, builtin_parser
from hcc.sema.constexpr import try_eval_const
from hcc.sema.conversions import new_cast


@builtin_parser("__builtin_alloca")
def _parse_alloca(self, tok):
    args = [new_cast(a, T.TY_ULONG) for a in self._args(1)]
    return self._call(tok, "alloca", args, T.pointer_to(T.TY_VOID))


@builtin_parser("__builtin_alloca_with_align")
def _parse_alloca_with_align(self, tok):
    """__builtin_alloca_with_align(size, align_in_bits): align must be a constant.
    The second argument of the lowered call is the alignment in bytes."""
    size, align = self._args(2)
    align_val = try_eval_const(align)
    byte_align = (align_val or 0) // 8
    if byte_align <= 0 or byte_align & (byte_align - 1):
        self.error("__builtin_alloca_with_align: alignment must be a power of two (in bits)")
    args = [new_cast(size, T.TY_ULONG), N.IntLit(tok.loc, byte_align, ty=T.TY_ULONG)]
    return self._call(tok, "alloca_align", args, T.pointer_to(T.TY_VOID))


@builtin_codegen("alloca")
def _gen_alloca(self, node) -> None:
    self.gen_expr(node.args[0])
    alloca_rax(self)


@builtin_codegen("alloca_align")
def _gen_alloca_align(self, node) -> None:
    """__builtin_alloca_with_align: over-allocate by align-1, then round up."""
    align = node.args[1].value
    self.gen_expr(node.args[0])
    self.emit(f"add ${align - 1}, %rax")
    alloca_rax(self)
    self.emit(f"add ${align - 1}, %rax")
    self.emit(f"mov ${-align}, %r10")
    self.emit("and %r10, %rax")


def alloca_call(self, node) -> None:
    """A plain call to `alloca(n)` (glibc's alloca.h maps it here; codegen/calls.py)."""
    self.gen_expr(node.args[0])
    alloca_rax(self)


def _update_alloca_mark(self) -> None:
    """Lower the function's alloca mark to the new block address in %r11."""
    mark = self.fn.alloca_bottom
    if mark is None:
        return
    self.emit(f"mov {mark.offset}(%rbp), %r10")
    self.emit("cmp %r10, %r11")
    self.emit("cmovb %r11, %r10")
    self.emit(f"mov %r10, {mark.offset}(%rbp)")


def alloca_rax(self) -> None:
    """Allocate %rax bytes (rounded to 16) on the stack; result in %rax.

    The stack machine may have `self.depth` pushed temporaries live at
    (%rsp), so the logical base is rsp + depth*8. The block is carved
    just below that base (16-aligned, gcc-like). The temporaries are
    copied to just below the block (ascending, so overlap is safe since
    the destination is always lower), and the block's bottom becomes the
    new depth-0 base, so later pops never reach the block."""
    d = self.depth
    self.emit("add $15, %rax")
    self.emit("and $-16, %rax")
    self.emit("mov %rsp, %r11")
    self.emit(f"add ${d * 8}, %r11")       # r11 = logical base
    self.emit("sub %rax, %r11")
    self.emit("and $-16, %r11")            # r11 = block address (new base)
    _update_alloca_mark(self)
    for i in range(d):
        self.emit(f"mov {i * 8}(%rsp), %r10")
        self.emit(f"mov %r10, {i * 8 - d * 8}(%r11)")
    self.emit("mov %r11, %rax")
    self.emit(f"lea {-d * 8}(%r11), %rsp")
