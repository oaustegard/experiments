"""-fplan9-extensions: implicit pointer conversion to an anonymous member.

With the flag, a pointer to a struct/union converts implicitly to a pointer to
one of its anonymous members (`struct S { char b; struct A; }` makes `struct S *`
usable as `struct A *`), and the pointer is adjusted by the member's offset.

plan9_pointer_conv(parser, expr, dst_ty) -> expr   (no-op unless applicable)
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.arith import new_add
from hcc.sema.conversions import new_cast
from hcc.sema.compat import is_compatible
from hcc.parser.structs import complete_struct


def _anon_offset(src, dst):
    """Byte offset of the (nested) anonymous member of `src` whose type is `dst`, or None."""
    for m in complete_struct(src).members:
        if m.name is not None or not T.is_struct_like(m.ty):
            continue
        if is_compatible(m.ty, dst):
            return m.offset
        sub = _anon_offset(m.ty, dst)
        if sub is not None:
            return m.offset + sub
    return None


def plan9_pointer_conv(parser, expr, dst_ty):
    opts = parser.opts
    if opts is None or not opts.plan9_extensions:
        return expr
    if dst_ty.kind != T.TypeKind.PTR or not T.is_struct_like(dst_ty.base):
        return expr
    src = parser.typed(expr).ty
    if src is None or src.kind != T.TypeKind.PTR or not T.is_struct_like(src.base):
        return expr
    if is_compatible(src.base, dst_ty.base):
        return expr
    off = _anon_offset(src.base, dst_ty.base)
    if off is None:
        return expr
    loc = expr.loc
    char_ptr = T.pointer_to(T.TY_CHAR)
    moved = new_add(new_cast(expr, char_ptr), N.IntLit(loc, off, ty=T.TY_LONG), loc)
    return new_cast(moved, dst_ty)
