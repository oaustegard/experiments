"""Block-scope `auto` function declarations need a definition (GNU nested functions).

NestedDeclMixin.check_nested_decls(scope): run when a block scope is left. Every
function declared with `auto` in that block (marked `nested_decl` by locals.py)
that is still only declared, not defined, is an error, whether or not it is used.
A later nested definition replaces the block entry, so it is not reported.
"""
from __future__ import annotations

from hcc import nodes as N


class NestedDeclMixin:

    def check_nested_decls(self, scope) -> None:
        for name, vs in scope.vars.items():
            fn = vs.var
            if isinstance(fn, N.Function) and getattr(fn, "nested_decl", False) \
                    and not fn.is_definition:
                self.error(f"nested function '{name}' declared but never defined", fn.loc)
