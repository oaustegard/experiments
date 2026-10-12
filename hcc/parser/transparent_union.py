"""GCC `__attribute__((transparent_union))` argument passing.

A call argument whose type matches one member of a transparent union parameter
is wrapped into a union value: the union temporary is initialised through that
member, and the call receives the union (same ABI as the member).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.compat import is_compatible
from hcc.sema.conversions import new_cast


def _pointee(ty):
    if ty.kind in (T.TypeKind.PTR, T.TypeKind.ARRAY):
        return ty.base
    return ty


def _member_accepts(mty, aty) -> bool:
    """Whether an argument of type `aty` may initialise a member of type `mty`."""
    if mty.is_complex or aty.is_complex or mty.is_vector or aty.is_vector:
        return False
    if T.is_struct_like(mty) or T.is_struct_like(aty):
        return T.is_struct_like(mty) and is_compatible(mty.unqual(), aty.unqual())
    m_ptr = mty.kind == T.TypeKind.PTR
    a_ptr = aty.kind in (T.TypeKind.PTR, T.TypeKind.ARRAY, T.TypeKind.FUNC)
    if m_ptr and a_ptr:
        mp = mty.base.unqual()
        ap = _pointee(aty).unqual()
        if mp.kind == T.TypeKind.VOID or ap.kind == T.TypeKind.VOID:
            return True
        return is_compatible(mp, ap)
    if m_ptr or a_ptr:
        return False
    return T.is_numeric(mty) and T.is_numeric(aty)


def apply_transparent_union(ty, attr, dattr):
    """Flag a union declared with transparent_union (on the body or after the declarator)."""
    flagged = any(a is not None and a.is_transparent_union for a in (attr, dattr))
    if not flagged or ty.kind != T.TypeKind.UNION or ty.is_transparent_union:
        return ty
    ty = ty.copy()
    ty.is_transparent_union = True
    return ty


def wrap_transparent_arg(parser, pty, arg, loc):
    """Return the union-typed argument for `arg` if `pty` is a transparent union, else None."""
    if pty.kind != T.TypeKind.UNION or not pty.is_transparent_union or arg.ty is None:
        return None
    member = next((m for m in pty.members if _member_accepts(m.ty, arg.ty)), None)
    if member is None:
        return None
    tmp = parser.new_temp(pty, loc)
    ref = parser.typed(N.VarRef(loc, tmp))
    target = parser.typed(N.Member_(loc, ref, member, ty=member.ty))
    init = parser.typed(N.Assign(loc, target, new_cast(arg, member.ty)))
    return parser.typed(N.Comma(loc, init, parser.typed(N.VarRef(loc, tmp)), lvalue=False))
