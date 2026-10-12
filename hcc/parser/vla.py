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

import copy

from hcc import nodes as N
from hcc import typesys as T
from hcc.parser.dyn_members import dyn_struct_size_expr
from hcc.parser.structs import is_dyn_member_type
from hcc.sema.conversions import new_cast


def _has_vla(ty) -> bool:
    """True when ty is a VLA, or a pointer/array chain ending in one."""
    while ty is not None and ty.kind in (T.TypeKind.PTR, T.TypeKind.ARRAY):
        ty = ty.base
    if T.is_dyn_struct(ty):
        return True
    return ty is not None and ty.kind == T.TypeKind.VLA and ty.vla_len is not None


def _rebind_struct(self, ty, loc):
    """Copy of struct `ty` whose run-time-sized members are bound (their VLA
    lengths evaluated into hidden locals). Returns (copy, statements)."""
    new = ty.copy()
    members = []
    pre = []
    for m in ty.members:
        if is_dyn_member_type(m.ty):
            mty, mpre = self.vla_bind(m.ty, loc)
            pre.extend(mpre)
            m = copy.copy(m)
            m.ty = mty
            new.dyn_tail = mty
        members.append(m)
    new.members = members
    return new, pre


def vla_slot_type(ty):
    """Type of the frame slot of a run-time-sized object: holds an address."""
    t = ty.copy()
    t.size = 8              # the real alignment is kept (sizeof reads the members)
    return t


class VlaMixin:

    def vla_bind(self, ty, loc):
        if T.is_dyn_struct(ty):
            return _rebind_struct(self, ty, loc)
        if ty.kind == T.TypeKind.ARRAY and _has_vla(ty.base):
            base, pre = self.vla_bind(ty.base, loc)
            new = ty.copy()
            new.base = base
            return new, pre
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
        if T.is_dyn_struct(ty):
            return dyn_struct_size_expr(self, ty, loc)
        if ty.kind == T.TypeKind.ARRAY and _has_vla(ty.base):
            elem = self.vla_size_expr(ty.base, loc)
            return self.typed(N.Binary(loc, "*", self.typed(N.IntLit(loc, ty.array_len, ty=T.TY_ULONG)), elem))
        if ty.kind != T.TypeKind.VLA:
            return self.typed(N.IntLit(loc, ty.size, ty=T.TY_ULONG))
        if ty.vla_len is None and self.in_proto_params:
            return self.typed(N.IntLit(loc, 0, ty=T.TY_ULONG))   # `[*]` in a prototype: unevaluated
        if ty.vla_len is None:
            # `[*]` is only legal in prototype scope, where a size is never
            # evaluated (e.g. `int c[sizeof(*b)]` after `int b[*][*]`).
            if getattr(self, "in_proto_params", False):
                return self.typed(N.IntLit(loc, 0, ty=T.TY_ULONG))
            self.error("invalid use of array with unspecified bounds", loc)
        length = new_cast(ty.vla_len, T.TY_ULONG)
        elem = self.vla_size_expr(ty.base, loc)
        return self.typed(N.Binary(loc, "*", length, elem))
