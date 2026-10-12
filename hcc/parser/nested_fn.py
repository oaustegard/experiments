"""Block-scope function definitions (GNU nested functions).

NestedFnMixin.nested_function(ty, name, attr, dattr): parses `T f(...) { ... }`
inside a block. The function is hoisted to file scope under a unique symbol
(`outer.f.N`) and records its enclosing Function in `outer`. Its body sees the
enclosing block's names; codegen reaches outer locals through a static link
(see codegen/nested.py).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.compat import is_compatible


class NestedFnMixin:

    def nested_function(self, ty, name, attr, dattr) -> None:
        outer = self.fn
        prior = self.find_var(name.text)
        if prior is not None and isinstance(prior.var, N.Function) \
                and not is_compatible(prior.var.ty, self.block_func_composite(ty, name.text)):
            self.error(f"conflicting types for '{name.text}'", name.loc)
        self.anon_count += 1
        fn = N.Function(name.text, ty, loc=name.loc)
        fn.is_static = True
        fn.is_inline |= attr.is_inline
        fn.is_noreturn |= attr.is_noreturn or dattr.is_noreturn
        fn.asm_name = f"{outer.symbol}.{name.text}.{self.anon_count}"
        fn.outer = outer
        self.push_scope_entry(name.text).var = fn
        self.objs.append(fn)
        saved = (self.fn, self.locals, self.label_names, self.goto_names)
        self.locals = []
        try:
            self.function_definition(fn, ty)
            # gotos to labels not in this body belong to the enclosing function
            pending = [g for g in self.goto_names if g.text not in self.label_names]
        finally:
            self.fn, self.locals, self.label_names, self.goto_names = saved
        self.goto_names.extend(pending)
