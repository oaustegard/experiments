"""Block-scope function declarations have external linkage (C11 6.2.2p5).

GCC: `void f(void) { long z(); }` declares z for the whole translation unit,
though the name is only lexically visible inside f. A later call `z()` in
another function is therefore not an implicit declaration: the implicit
`int z()` conflicts with the external declaration, which is an error
("incompatible implicit declaration of function 'z'").

Only a block-scope declaration that created the file-scope entry is marked
`block_decl_only`; a later file-scope declaration clears the mark (see
toplevel.declare_function).
"""
from hcc.sema.compat import is_compatible
from hcc import typesys as T


class BlockFuncVisibilityMixin:
    def check_block_func_visible(self, name, fn, tok, is_call) -> None:
        if not is_call or not getattr(fn, "block_decl_only", False):
            return
        for scope in self.scopes[1:]:
            vs = scope.vars.get(name)
            if vs is not None and vs.var is fn:
                return                      # lexically visible: a normal reference
        implicit = T.func_type(T.TY_INT)
        implicit.has_proto = False
        if not is_compatible(implicit, fn.ty):
            self.error(f"incompatible implicit declaration of function '{name}'", tok)
