"""__builtin_tgmath(f1, ..., fn, v1, ..., vk): GCC's type-generic call (C11 7.22).

The leading arguments are the candidate functions (function designators); the
last k arguments are the values, where k is the parameter count of the candidates.
The common type of the values is: complex if any value is complex; otherwise the
widest real floating type. Integer values count as double, and a real floating
value wins a width tie against them. The first candidate whose parameters all have
exactly that type is called, with the values converted to it.
"""
from __future__ import annotations

from hcc import typesys as T
from hcc.parser import complex as CPX
from hcc.typesys import TypeKind

_FLOAT_KINDS = (TypeKind.FLOAT16, TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LDOUBLE, TypeKind.FLOAT128)


def _func_type(node):
    """FUNC type of a function designator or function pointer, else None."""
    t = node.ty
    if t.kind == TypeKind.PTR and t.base is not None and t.base.kind == TypeKind.FUNC:
        t = t.base
    return t if t.kind == TypeKind.FUNC else None


def _fp_elem(t):
    return t.members[0].ty if t.is_complex else t


def _is_fp(t) -> bool:
    return t.is_complex or t.kind in _FLOAT_KINDS


def _same_fp(a, b) -> bool:
    """Strict type match: same complexness, same floating kind and same _FloatN spelling."""
    if not (_is_fp(a) and _is_fp(b)) or a.is_complex != b.is_complex:
        return False
    x, y = _fp_elem(a), _fp_elem(b)
    return x.kind == y.kind and x.floatn == y.floatn


def common_type(p, loc, vals):
    """Common floating type of the value arguments (see the module docstring)."""
    has_cplx = False
    best = None
    best_from_int = False
    for v in vals:
        t = v.ty
        if t.is_complex:
            has_cplx = True
            e, from_int = _fp_elem(t), False
        elif t.kind in _FLOAT_KINDS:
            e, from_int = t, False
        elif T.is_integer(t):
            e, from_int = T.TY_DOUBLE, True
        else:
            p.error(f"invalid argument of type '{t}' to __builtin_tgmath", loc)
        if best is None or e.size > best.size or \
                (e.size == best.size and best_from_int and not from_int):
            best, best_from_int = e, from_int
    return T.complex_of(best) if has_cplx else best


def tgmath_call(p, tok):
    """Parses the arguments after `__builtin_tgmath(` (the caller consumes `)`)."""
    args = []
    while True:
        args.append(p.typed(p.assign()))
        if not p.accept(","):
            break
    n = 0
    while n < len(args) and _func_type(args[n]) is not None:
        n += 1
    if n == 0 or n == len(args):
        p.error("__builtin_tgmath needs function designators followed by values", tok)
    funcs, vals = args[:n], args[n:]
    k = len(vals)
    common = common_type(p, tok.loc, vals)
    for fn in funcs:
        fty = _func_type(fn)
        if not fty.has_proto or len(fty.params) != k or fty.is_variadic:
            continue
        if all(_same_fp(prm, common) for prm in fty.params):
            conv = [CPX.convert(p, v, common, tok.loc) for v in vals]
            return p.build_call(fn, conv, tok.loc)
    p.error(f"no function of __builtin_tgmath matches the common type '{common}'", tok)
