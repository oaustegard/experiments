"""Struct with a variably modified member as a function result, and assignment to one.

Such a struct (`typesys.is_dyn_struct`) has a run-time size. As for by-value
arguments (dyn_struct_args.py), the value of an expression of this type is the
ADDRESS of its bytes, and the SysV MEMORY convention carries the storage:

* caller: `dyn_ret_buffer` allocates the result block (a VLA-style alloca, saved
  in a pointer temp `ret_buffer`). The call node carries the allocation as
  `dyn_alloc`; codegen emits it before the enclosing statement, when %rsp is at
  statement depth, and passes the temp's value as the hidden first argument.
* callee: `dyn_return` gives the Return statement the run-time size; codegen
  copies the value into the hidden pointer and returns that pointer.

dyn_struct_assign(parser, lhs, rhs, loc) -> Expr   (memcpy of the run-time size)
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.parser.dyn_struct_args import _memcpy


def dyn_ret_buffer(parser, ret_ty, loc):
    """(pointer temp, VLAAlloc) for a call returning a struct of type `ret_ty`."""
    ptr_ty = T.pointer_to(T.TY_VOID)
    size = parser.vla_size_expr(ret_ty, loc)
    tmp = parser.new_temp(ptr_ty, loc)
    # Not appended to block_vla_saves: the result is only live within its statement, and
    # codegen chains the allocations of one statement (see codegen/dyn_struct_ret.py).
    alloc = N.VLAAlloc(loc, tmp, parser.vla_block_slot(), size)
    return tmp, alloc


def dyn_return(parser, expr, ret_ty, loc):
    """`return expr;` in a function returning a struct with a variably modified member."""
    if not T.is_dyn_struct(expr.ty):
        parser.error("unsupported: returning a non-variably-modified struct from a function "
                     "returning a struct with a variably modified member", loc)
    return N.Return(loc, expr, size=parser.vla_size_expr(expr.ty, loc))


def dyn_struct_assign(parser, lhs, rhs, loc):
    """`lhs = rhs` on a struct with a variably modified member: memcpy of the run-time size.
    The value is the address of lhs (like every struct value)."""
    ptr_ty = T.pointer_to(T.TY_VOID)
    if not T.is_dyn_struct(rhs.ty):
        parser.error("unsupported: assignment of a non-variably-modified struct to a struct "
                     "with a variably modified member", loc)
    size = parser.vla_size_expr(rhs.ty, loc)
    dst = parser.new_temp(ptr_ty, loc)
    addr = N.Cast(loc, lhs)              # a struct value is the address of its bytes
    addr.ty = ptr_ty
    store = parser.typed(N.Assign(loc, parser.typed(N.VarRef(loc, dst)), addr))
    src = N.Cast(loc, rhs)
    src.ty = ptr_ty
    fn = _memcpy(parser, loc)
    call = parser.build_call(parser.typed(N.VarRef(loc, fn)),
                             [parser.typed(N.VarRef(loc, dst)), src, size], loc)
    body = N.Block(loc, [N.ExprStmt(loc, store), N.ExprStmt(loc, call),
                         N.ExprStmt(loc, parser.typed(N.VarRef(loc, dst)))])
    out = parser.typed(N.StmtExpr(loc, body))
    out.ty = lhs.ty
    return out
