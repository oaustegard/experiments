"""Static links for GNU nested functions.

A nested Function (`fn.outer` set) gets the %rbp of its enclosing function in
%r10 at every direct call; its prologue saves that in `fn.chain_var`. Outer
locals are reached by following those slots from the current frame.
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc.typesys import TypeKind, is_dyn_struct


class NestedMixin:

    def nested_callee(self, node):
        """The nested Function called directly by `node`, or None."""
        f = node.func
        if type(f) is N.VarRef and isinstance(f.var, N.Function) and f.var.outer is not None:
            return f.var
        return None

    def is_outer_local(self, var) -> bool:
        fn = self.fn
        return fn is not None and fn.outer is not None and var.is_local and \
            var not in fn.locals and var not in fn.params

    def frame_of(self, owner) -> None:
        """Leave the %rbp of enclosing function `owner` (self.fn or an ancestor) in %r11."""
        self.emit("mov %rbp, %r11")
        cur = self.fn
        while cur is not owner:
            if cur is None or cur.chain_var is None:
                diag.error(self.fn.loc, f"cannot reach enclosing frame of '{owner.name}'")
            self.emit(f"mov {cur.chain_var.offset}(%r11), %r11")
            cur = cur.outer

    def outer_var_addr(self, var) -> None:
        cur = self.fn.outer
        while cur is not None and var not in cur.locals and var not in cur.params:
            cur = cur.outer
        if cur is None:
            diag.error(var.loc, f"'{var.name}' is not in an enclosing function")
        self.frame_of(cur)
        if var.ty.kind == TypeKind.VLA or is_dyn_struct(var.ty):
            self.emit(f"mov {var.offset}(%r11), %rax")
        else:
            self.emit(f"lea {var.offset}(%r11), %rax")

    def emit_static_link(self, callee) -> None:
        self.frame_of(callee.outer)
        self.emit("mov %r11, %r10")
