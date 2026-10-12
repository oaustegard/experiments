"""Simple assignment compatibility for struct/union operands (C11 6.5.16.1).

A struct or union may only be assigned from a compatible struct or union, and
no scalar may be assigned from a struct or union.
"""
from __future__ import annotations

from hcc.sema.compat import is_compatible
from hcc.typesys import is_dyn_struct, is_struct_like


def check_struct_assign(parser, lhs_ty, rhs_ty, tok) -> None:
    if lhs_ty is None or rhs_ty is None:
        return
    if lhs_ty.is_complex or rhs_ty.is_complex or is_dyn_struct(lhs_ty) or is_dyn_struct(rhs_ty):
        return
    l_agg, r_agg = is_struct_like(lhs_ty), is_struct_like(rhs_ty)
    if not (l_agg or r_agg):
        return
    if l_agg and r_agg and is_compatible(lhs_ty.unqual(), rhs_ty.unqual()):
        return
    parser.error("incompatible types when assigning to type from type", tok)
