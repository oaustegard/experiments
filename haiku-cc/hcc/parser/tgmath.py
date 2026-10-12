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


def _same_fp(a, b, strict=True) -> bool:
    """Type match: same floating kind and _FloatN spelling; with `strict`, also the same complexness."""
    if not (_is_fp(a) and _is_fp(b)) or (strict and a.is_complex != b.is_complex):
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
            if T.is_integer(e):  # complex integers count as complex double
                e, from_int = T.TY_DOUBLE, True
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


def _generic_positions(funcs, k):
    """Argument positions whose parameter is floating in some candidate (type-generic, C11 7.22p2).

    Non-floating parameters (e.g. the `int` of `ldexp`-like candidates) are not type-generic:
    their arguments neither set the common type nor have to match it.
    """
    pos = set()
    for fn in funcs:
        fty = _func_type(fn)
        if fty is None or not fty.has_proto or len(fty.params) != k:
            continue
        for i, prm in enumerate(fty.params):
            if _is_fp(prm):
                pos.add(i)
    return pos


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
    generic = _generic_positions(funcs, k)
    gvals = [vals[i] for i in sorted(generic)] if generic else vals
    common = common_type(p, tok.loc, gvals)
    fn = _pick(funcs, k, sorted(generic) or list(range(k)), common)
    if fn is None:
        p.error(f"no function of __builtin_tgmath matches the common type '{common}'", tok)
    fty = _func_type(fn)
    conv = [CPX.convert(p, v, fty.params[i], tok.loc) if i in generic else v
            for i, v in enumerate(vals)]
    return p.build_call(fn, conv, tok.loc)


def _widened_size(fty, idx, common):
    """Element size of the candidate's type-generic params if they are all one floating
    type at least as wide as `common`, else None."""
    sizes = set()
    for i in idx:
        prm = fty.params[i]
        if not _is_fp(prm):
            return None
        sizes.add(_fp_elem(prm).size)
    if len(sizes) != 1:
        return None
    size = sizes.pop()
    return size if size >= _fp_elem(common).size else None


def _pick(funcs, k, idx, common):
    """The candidate for `common`: an exact match first (complexness, then real type only);
    failing that, the narrowest candidate at least as wide (C11 7.22p2: a float value with
    no float candidate uses the double one)."""
    cands = []
    for fn in funcs:
        fty = _func_type(fn)
        if fty.has_proto and len(fty.params) == k and not fty.is_variadic:
            cands.append((fn, fty))
    for strict in (True, False):
        for fn, fty in cands:
            if all(_same_fp(fty.params[i], common, strict) for i in idx):
                return fn
    best = None
    for fn, fty in cands:
        size = _widened_size(fty, idx, common)
        if size is not None and (best is None or size < best[0]):
            best = (size, fn)
    return best[1] if best is not None else None
