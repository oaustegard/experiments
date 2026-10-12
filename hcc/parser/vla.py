"""Variable length array types (block scope).

VlaMixin.vla_bind(ty, loc) -> (Type, list[Stmt])
    Evaluates the length expressions of a VLA type once, at its declaration
    point, into hidden ulong locals. Returns a copy of the type whose
    vla_len refers to those locals, plus the statements doing the evaluation.
VlaMixin.vla_size_expr(ty, loc) -> Expr
    Runtime byte size (ulong expression) of a type; used by sizeof.
Only typedefs of VLA types are supported so far (no VLA objects yet).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.conversions import new_cast


class VlaMixin:

    def vla_bind(self, ty, loc):
        if ty.kind != T.TypeKind.VLA or ty.vla_len is None:
            return ty, []
        pre = []
        new = ty.copy()
        # C evaluates the outermost dimension first, then the inner ones.
        tmp = self.new_temp(T.TY_ULONG, loc)
        store = self.typed(N.Assign(loc, self.typed(N.VarRef(loc, tmp)),
                                    new_cast(ty.vla_len, T.TY_ULONG)))
        pre.append(N.ExprStmt(loc, store))
        new.vla_len = self.typed(N.VarRef(loc, tmp))
        base, base_pre = self.vla_bind(ty.base, loc)
        new.base = base
        return new, pre + base_pre

    def vla_size_expr(self, ty, loc):
        if ty.kind != T.TypeKind.VLA:
            return self.typed(N.IntLit(loc, ty.size, ty=T.TY_ULONG))
        if ty.vla_len is None:
            self.error("invalid use of array with unspecified bounds", loc)
        length = new_cast(ty.vla_len, T.TY_ULONG)
        elem = self.vla_size_expr(ty.base, loc)
        return self.typed(N.Binary(loc, "*", length, elem))
