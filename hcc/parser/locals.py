"""Block-scope declarations.

LocalDeclMixin.local_declaration() -> list[Stmt]
Handles typedefs, function prototypes, extern and static locals, automatic
variables with initializers (lowered to ExprStmt), __auto_type.
VLAs: TODO (currently a parse error).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.parser.scope import VarAttr
from hcc import typesys as T


class LocalDeclMixin:

    def local_declaration(self) -> list:
        attr = VarAttr()
        basety = self.declspec(attr)
        items = []
        first = True
        while not self.accept(";"):
            if not first:
                self.expect(",")
            first = False
            if basety is None:                       # __auto_type x = expr;
                name = self.expect_ident()
                self.expect("=")
                e = self.typed(self.assign())
                ty = e.ty
                if ty.kind == T.TypeKind.ARRAY:
                    ty = T.pointer_to(ty.base)
                var = self.new_lvar(name.text, ty, name.loc)
                items.append(N.ExprStmt(name.loc, self.typed(N.Assign(name.loc, self.typed(N.VarRef(name.loc, var)), e))))
                continue
            ty, name = self.declarator(basety)
            dattr = VarAttr()
            self.parse_attributes(dattr)
            self.parse_asm_label(dattr)
            if name is None:
                self.error("identifier expected in declaration")
            if attr.is_typedef:
                self.push_scope_entry(name.text).type_def = ty
                continue
            if ty.kind == T.TypeKind.FUNC:
                fn = self.declare_function(ty, name, attr, dattr)
                self.push_scope_entry(name.text).var = fn
                continue
            if attr.is_extern:
                var = self.declare_global_var(ty, name, attr, dattr)
                self.push_scope_entry(name.text).var = var
                continue
            if attr.is_static:
                self.anon_count += 1
                var = N.Var(f"{name.text}.{self.anon_count}", ty, is_static=True, loc=name.loc,
                            is_definition=True, align=max(attr.align, dattr.align) or ty.align,
                            is_tls=attr.is_tls)
                self.objs.append(var)
                self.push_scope_entry(name.text).var = var
                if self.accept("="):
                    self.global_initializer(var)
                elif not ty.is_complete and ty.kind != T.TypeKind.ARRAY:
                    self.error(f"storage size of '{name.text}' isn't known", name)
                continue
            if ty.kind == T.TypeKind.VOID:
                self.error(f"variable '{name.text}' declared void", name)
            if ty.kind == T.TypeKind.VLA:
                self.error("variable length arrays are not supported yet", name)
            var = self.new_lvar(name.text, ty, name.loc)
            al = max(attr.align, dattr.align)
            if al:
                var.align = max(al, ty.align)
            if self.accept("="):
                e = self.local_initializer(var)
                items.append(N.ExprStmt(name.loc, e))
            if not var.ty.is_complete or var.ty.size < 0:
                self.error(f"storage size of '{name.text}' isn't known", name)
        return items
