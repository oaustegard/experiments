"""Value of a const-qualified pointer object with a static initializer (C23 constexpr
pointers): an address constant `(label, addend)` or a plain number. None if not constant.

Used by sema/constexpr.py (`_eval`, VarRef of pointer type)."""
from __future__ import annotations


def const_ptr_value(var):
    """(label, addend) for an address constant, (None, raw) for a number, or None."""
    data = getattr(var, "init_data", None)
    if data is None or getattr(var, "is_local", False) or not var.ty.is_const \
            or var.ty.is_volatile:
        return None
    if not data.relocs:
        return None, int.from_bytes(bytes(data.data[:8]), "little")
    if len(data.relocs) != 1:
        return None
    r = data.relocs[0]
    if r.offset != 0 or r.minus or r.size != 8:
        return None
    return r.label, r.addend
