"""Old-style (K&R) function definitions must agree with a prior prototype
(C11 6.9.1p7, 6.7.6.3p15): each parameter type of the definition, after
default argument promotions, must be compatible with the prototype's type.
"""
from __future__ import annotations

from hcc import typesys as T
from hcc.sema.compat import is_compatible

_PROMOTED = (T.TypeKind.CHAR, T.TypeKind.SHORT, T.TypeKind.BOOL, T.TypeKind.FLOAT)


def _promote(t):
    """Default argument promotion of a K&R declared type: char/short/bool
    become int, float becomes double (_FloatN types are left alone)."""
    if t.is_floatn:
        return t
    if t.kind in (T.TypeKind.CHAR, T.TypeKind.SHORT, T.TypeKind.BOOL):
        return T.TY_INT
    if t.kind == T.TypeKind.FLOAT:
        return T.TY_DOUBLE
    return t


def _param_ok(x, y) -> bool:
    return is_compatible(x, y) and _unproto_ok(x, y)


def _unproto_ok(a, b) -> bool:
    """Walk pointer chains; where exactly one side is an unprototyped function,
    the prototyped side must not have parameters that default promotion changes."""
    if a.kind == T.TypeKind.PTR and b.kind == T.TypeKind.PTR:
        return _unproto_ok(a.base, b.base)
    if a.kind == T.TypeKind.FUNC and b.kind == T.TypeKind.FUNC and a.has_proto != b.has_proto:
        proto = a if a.has_proto else b
        for p in proto.params:
            if p.kind in _PROMOTED and not p.is_floatn:
                return False
    return True


def knr_param_index(proto_fty, decl_tys):
    """Index of the first K&R parameter whose declared type (before default
    promotion, so `float a;` is still float) does not match the prior prototype
    `proto_fty`, or None when they agree.  The K&R type is compared after
    default argument promotion (`unsigned char c;` matches an `int` prototype); either form may match."""
    if not proto_fty.has_proto or len(proto_fty.params) != len(decl_tys):
        return None
    for i, (x, y) in enumerate(zip(proto_fty.params, decl_tys)):
        x = x.unqual()
        y = y.unqual()
        if _param_ok(x, y) or _param_ok(x, _promote(y)):
            continue
        return i
    return None
