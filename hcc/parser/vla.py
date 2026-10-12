"""Variable length array types (block scope).

VlaMixin.vla_bind(ty, loc) -> (Type, list[Stmt])
    Evaluates the length expressions of a VLA type once, at its declaration
    point, into hidden ulong locals. Returns a copy of the type whose
    vla_len refers to those locals, plus the statements doing the evaluation.
VlaMixin.vla_size_expr(ty, loc) -> Expr
    Runtime byte size (ulong expression) of a type; used by sizeof.
VlaMixin.vla_prefix(pre, expr, loc) -> Expr
    Wraps expr so the length evaluations in pre run first (casts, compound
    literals and locals of pointer-to-VLA type, e.g. `(int (*)[++i])p`).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.conversions import new_cast


def _has_vla(ty) -> bool:
    """True when ty is a VLA, or a pointer/array chain ending in one."""
    while ty is not None and ty.kind in (T.TypeKind.PTR, T.TypeKind.ARRAY):
        ty = ty.base
    return ty is not None and ty.kind == T.TypeKind.VLA and ty.vla_len is not None


class VlaMixin:

    def vla_bind(self, ty, loc):
        if ty.kind == T.TypeKind.PTR and _has_vla(ty.base):
            base, pre = self.vla_bind(ty.base, loc)
            new = ty.copy()
            new.base = base
            return new, pre
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

    def vla_prefix(self, pre, expr, loc):
        for st in reversed(pre):
            expr = self.typed(N.Comma(loc, st.expr, expr, lvalue=getattr(expr, "lvalue", False)))
        return expr

    def vla_block_slot(self):
        """Var whose slot holds the %rsp the next VLA of the innermost block
        is carved below: the most recent VLA of that block (its slot holds its
        data address = its %rsp), else the hidden block-entry slot (created on
        first use; see compound_stmt_body). saves[0] stays the block entry."""
        saves = self.block_vla_saves
        if not saves:
            saves.append(self.new_temp(T.TY_ULONG))
        return saves[-1]

    def vla_size_expr(self, ty, loc):
        if ty.kind != T.TypeKind.VLA:
            return self.typed(N.IntLit(loc, ty.size, ty=T.TY_ULONG))
        if ty.vla_len is None:
            self.error("invalid use of array with unspecified bounds", loc)
        length = new_cast(ty.vla_len, T.TY_ULONG)
        elem = self.vla_size_expr(ty.base, loc)
        return self.typed(N.Binary(loc, "*", length, elem))
