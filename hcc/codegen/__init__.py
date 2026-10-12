"""Code generation: nodes.TranslationUnit -> AT&T assembly text.

    generate(tu, opts=None) -> str

CodeGen is assembled from one mixin per concern (CONTRACT section 2/7): every
`*Mixin` class defined in a module of this package, found by hcc.registry.
Stack-machine design: expressions leave their value in %rax / %xmm0 / st(0).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.codegen.context import CodeGenBase
from hcc.codegen.lazy_inline import lazy_functions, needed_lazy
from hcc.registry import assemble_mixins


# Every `*Mixin` class in a module of hcc/codegen/ is a base (hcc/registry.py):
# a new concern is a new file, with no import or base list to edit here.
class CodeGen(*assemble_mixins("hcc.codegen", CodeGenBase)):

    def run(self) -> str:
        self.directive(f'.file "{getattr(self.opts, "source_name", "") or "hcc"}"')
        lazy = lazy_functions(self.tu)
        lazy_ids = {id(f) for f in lazy}
        needed = needed_lazy(self.tu, lazy) if lazy else set()
        for obj in self.tu.objs:
            if id(obj) in lazy_ids:
                continue          # emitted below, if referenced
            if isinstance(obj, N.Asm):
                self.emit_toplevel_asm(obj)
            elif isinstance(obj, N.Function):
                if obj.is_definition:
                    self.emit_function(obj)
                elif obj.alias_of:
                    self.emit_alias(obj)
                elif obj.ifunc_of:
                    self.emit_ifunc(obj)
                elif obj.is_weak:
                    # Weak declaration without a definition here: the symbol
                    # is an undefined weak reference (resolves to 0 if absent).
                    self.directive(f".weak {obj.symbol}")
            elif isinstance(obj, N.Var):
                if obj.alias_of and obj.init_data is None:
                    # `T A __attribute__((alias("a")))`: the symbol names a's storage
                    self.emit_alias(obj)
                elif obj.is_definition and not obj.is_extern:
                    self.emit_data(obj)
                elif obj.alias_of and not obj.is_definition:
                    self.emit_alias(obj)
                elif obj.is_weak and not obj.is_definition:
                    self.directive(f".weak {obj.symbol}")
        for fn in lazy:
            if id(fn) in needed:
                fn.is_weak = True     # no inline-only definition is unique across TUs
                self.emit_function(fn)
        self.emit_init_fini()
        self.directive('.section .note.GNU-stack,"",@progbits')
        return "\n".join(self.out) + "\n"

    def emit_alias(self, fn) -> None:
        if not fn.is_static:
            self.directive(f".globl {fn.symbol}")
        if fn.is_weak:
            self.directive(f".weak {fn.symbol}")
        self.directive(f".set {fn.symbol}, {fn.alias_of}")

    def emit_ifunc(self, fn) -> None:
        """GNU indirect function: the symbol's value is the resolver's address."""
        if not fn.is_static:
            self.directive(f".globl {fn.symbol}")
        if fn.visibility in ("hidden", "internal", "protected"):
            self.directive(f".{fn.visibility} {fn.symbol}")
        self.directive(f".type {fn.symbol}, @gnu_indirect_function")
        self.directive(f".set {fn.symbol}, {fn.ifunc_of}")


def generate(tu, opts=None) -> str:
    return CodeGen(tu, opts).run()
