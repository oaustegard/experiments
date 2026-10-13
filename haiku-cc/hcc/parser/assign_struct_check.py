"""Simple assignment compatibility for struct/union operands (C11 6.5.16.1).

A struct or union may only be assigned from a compatible struct or union.
For a return, a struct/union never converts to or from a scalar either.
Assignment between a struct/union and a scalar is left to sema
(`sema/typing.py:_check_struct_scalar_mix`), which reports it at the
[sema] stage; reporting it here would shadow that check.
"""
from __future__ import annotations

from hcc.sema.compat import is_compatible
from hcc.typesys import is_dyn_struct, is_struct_like, type_str


def check_struct_assign(parser, lhs_ty, rhs_ty, tok, *, site: str = "assign") -> None:
    """Reject a struct/union operand mismatch; `site` is 'assign' or 'return'."""
    if lhs_ty is None or rhs_ty is None:
        return
    if lhs_ty.is_complex or rhs_ty.is_complex or is_dyn_struct(lhs_ty) or is_dyn_struct(rhs_ty):
        return
    l_agg, r_agg = is_struct_like(lhs_ty), is_struct_like(rhs_ty)
    if not (l_agg or r_agg):
        return
    if l_agg and r_agg and is_compatible(lhs_ty.unqual(), rhs_ty.unqual()):
        return
    if site == "assign" and not (l_agg and r_agg):
        return
    if site == "return":
        msg = (f"incompatible types when returning type '{type_str(rhs_ty)}' "
               f"but '{type_str(lhs_ty)}' was expected")
    else:
        msg = (f"incompatible types when assigning to type '{type_str(lhs_ty)}' "
               f"from type '{type_str(rhs_ty)}'")
    parser.error(msg, tok)
