"""Constant folding for _Complex values at file scope (static initializers).

The lowering in parser/complex.py builds temporaries, which cannot be
serialized into .data.  Instead, every lowered complex node that has a
compile-time value carries `cval = (re, im)` (Python int/float) and the
file-scope initializer becomes a struct initializer with one constant leaf
per part.  Values are rounded to the element type when stored, so a
`_Complex float` constant holds float-precision parts.

Non-constant complex initializers at file scope are a clear error.
"""
from __future__ import annotations

import struct

from hcc import nodes as N
from hcc.typesys import TypeKind, is_flonum, is_integer


def value_of(node):
    """(re, im) of a constant operand, or None."""
    cv = getattr(node, "cval", None)
    if cv is not None:
        return cv
    if isinstance(node, N.IntLit):
        return (node.value, 0)
    if isinstance(node, N.FloatLit):
        return (node.value, 0.0)
    if isinstance(node, N.Unary) and node.op in ("-", "+") and not node.ty.is_complex:
        cv = value_of(node.operand)
        if cv is not None:
            return (store_value(node.ty, -cv[0] if node.op == "-" else cv[0]), 0)
    if isinstance(node, N.Cast) and not node.ty.is_complex and (is_flonum(node.ty) or is_integer(node.ty)):
        cv = value_of(node.operand)      # a real cast, or the real part of a complex constant
        if cv is not None:
            return (store_value(node.ty, cv[0]), 0)
    return None


def real_value(node):
    """Compile-time value of a real operand (literals and casts of them), or None."""
    cv = value_of(node)
    return None if cv is None else cv[0]


def store_value(base, v):
    """Round/convert a folded number to the element type `base`."""
    if is_integer(base):
        return int(v)
    if base.kind == TypeKind.FLOAT:
        return struct.unpack("<f", struct.pack("<f", float(v)))[0]
    return float(v)


def _div(a, b, integral):
    if integral:
        if b == 0:
            return None
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b >= 0) else -q
    if b == 0:
        return None
    return a / b


def fold(op, a, b, base):
    """Fold `a op b` for complex/real constant parts; None if not foldable."""
    if op in ("==", "!="):
        eq = a[0] == b[0] and a[1] == b[1]
        return (1 if eq == (op == "==") else 0, 0)
    integral = is_integer(base)
    (ar, ai), (br, bi) = a, b
    if op == "+":
        re, im = ar + br, ai + bi
    elif op == "-":
        re, im = ar - br, ai - bi
    elif op == "*":
        re, im = ar * br - ai * bi, ar * bi + ai * br
    elif op == "/":
        d = br * br + bi * bi
        re = _div(ar * br + ai * bi, d, integral)
        im = _div(ai * br - ar * bi, d, integral)
        if re is None or im is None:
            return None
    else:
        return None
    return (store_value(base, re), store_value(base, im))


def attach(node, cval, base):
    """Record the folded value on a lowered node (rounded to `base`)."""
    if cval is not None:
        node.cval = (store_value(base, cval[0]), store_value(base, cval[1]))
    return node


def set_global_init(p, init) -> None:
    """File-scope complex initializer: one constant leaf per part (struct children)."""
    loc = init.expr.loc
    cv = value_of(init.expr)
    if not init.ty.is_complex:
        # C11 6.3.1.7: complex -> real conversion keeps the real part, drops the imaginary part
        if not (is_flonum(init.ty) or is_integer(init.ty)):
            p.error("complex value used to initialize a non-arithmetic object", loc)
        if cv is None:
            p.error("initializer element is not constant", loc)
        real = cv[0] != 0 if init.ty.kind == TypeKind.BOOL else cv[0]
        v = store_value(init.ty, real)
        if is_flonum(init.ty):
            init.expr = p.typed(N.FloatLit(loc, float(v), ty=init.ty))
        else:
            init.expr = p.typed(N.IntLit(loc, int(v), ty=init.ty))
        return
    if cv is None:
        p.error("initializer element is not constant", loc)
    if not init.ty.is_complex:         # a real object takes the real part (GNU)
        if not (is_flonum(init.ty) or is_integer(init.ty)):
            p.error("complex value used to initialize a non-arithmetic object", loc)
        v = store_value(init.ty, cv[0])
        init.expr = p.typed(N.FloatLit(loc, float(v), ty=init.ty) if is_flonum(init.ty)
                            else N.IntLit(loc, int(v), ty=init.ty))
        return
    base = init.ty.base
    init.children = [N.Initializer(base), N.Initializer(base)]
    for child, v in zip(init.children, store_value_pair(base, cv)):
        if is_flonum(base):
            child.expr = p.typed(N.FloatLit(loc, float(v), ty=base))
        else:
            child.expr = p.typed(N.IntLit(loc, int(v), ty=base))
    init.expr = None


def store_value_pair(base, cv):
    return (store_value(base, cv[0]), store_value(base, cv[1]))

