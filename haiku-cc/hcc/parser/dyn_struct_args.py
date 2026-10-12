"""Passing a struct with a variably modified member by value (GNU extension).

Such a struct (`typesys.is_dyn_struct`) has a run-time size, so its bytes cannot
be placed in the fixed SysV argument slots. hcc passes it as one INTEGER
argument: the address of a fresh block-scope copy that the caller makes (so
the callee may modify its parameter freely). The callee's parameter is a
run-time-sized object whose frame slot holds that address (vla_slot_type).

dyn_struct_arg(parser, arg, loc) -> Expr   (pointer-typed copy address)
"""
from __future__ import annotations

from types import SimpleNamespace

from hcc import nodes as N
from hcc import typesys as T


def _memcpy(parser, loc):
    vs = parser.find_var("memcpy")
    if vs is not None and isinstance(vs.var, N.Function):
        return vs.var
    return parser.implicit_function("memcpy", SimpleNamespace(loc=loc), T.pointer_to(T.TY_VOID))


def dyn_struct_arg(parser, arg, loc):
    ptr_ty = T.pointer_to(T.TY_VOID)
    size = parser.vla_size_expr(arg.ty, loc)
    tmp = parser.new_temp(ptr_ty, loc)
    alloc = N.VLAAlloc(loc, tmp, parser.vla_block_slot(), size)
    parser.block_vla_saves.append(tmp)     # later VLAs of this block stack below the copy
    fn = _memcpy(parser, loc)
    src = N.Cast(loc, arg)                 # a struct value is the address of its bytes
    src.ty = ptr_ty
    dst = parser.typed(N.VarRef(loc, tmp))
    call = parser.build_call(parser.typed(N.VarRef(loc, fn)), [dst, src, size], loc)
    body = N.Block(loc, [alloc, N.ExprStmt(loc, call), N.ExprStmt(loc, parser.typed(N.VarRef(loc, tmp)))])
    return parser.typed(N.StmtExpr(loc, body))
