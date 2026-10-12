"""C99 _Complex lowering (GCC extension semantics).

A complex value is a two-member struct built by typesys.complex_of(): `re` at
offset 0 and `im` right after it, flagged `is_complex`.  Every operator is
lowered here, in the parser, into Comma / Assign / Member_ trees over anonymous
temporaries (Parser.new_temp), so codegen needs no complex-specific code and
the SysV struct ABI applies (_Complex float/double pass and return like
struct {float re, im} / {double re, im}).

Supported: + - * / == != unary -, __real__ / __imag__, imaginary constants
(1.5i, 2j), conversions between complex and real/complex types (assignment,
return, casts, scalar local initializers, binary operands).
Not supported (clear error): complex integers, complex long double (its ABI
differs from the struct ABI), complex in boolean context.

Division uses the textbook formula, falling back to libgcc __divdc3/__divsc3
(complex_div.py) when the quotient is NaN+iNaN (C Annex G).

    complex_decl_type(p, ty)      declspec: `_Complex` applied to a base type
    complex_literal(p, loc, v, cty)   imaginary constant -> complex temporary
    make_complex(p, loc, re, im)  __builtin_complex(re, im)
    binary(p, op, lhs, rhs, loc)  arithmetic / equality with a complex operand
    negate(p, operand, loc)       unary minus on a complex operand
    conjugate(p, operand, loc)    GCC `~z` on a complex operand
    part(p, operand, which, loc)  __real__ (0) / __imag__ (1) lvalue or rvalue
    convert(p, expr, ty, loc)     implicit/explicit conversion to ty
    builtin_complex(p, loc, args) __builtin_complex(re, im)
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.arith import new_binop
from hcc.parser import complex_const as CC
from hcc.parser.complex_div import recover_division
from hcc.sema.conversions import common_type, new_cast
from hcc.typesys import TY_DOUBLE, TypeKind, complex_of, is_flonum, is_integer

_RANK = {TypeKind.FLOAT16: 0, TypeKind.FLOAT: 1, TypeKind.DOUBLE: 2, TypeKind.LDOUBLE: 3,
         TypeKind.FLOAT128: 4}


def complex_decl_type(p, ty):
    """`_Complex T` (T defaults to double): the complex struct type for T."""
    if ty is None:
        ty = TY_DOUBLE
    if ty.is_complex:
        return ty
    if ty.kind not in _RANK and not (is_integer(ty) and ty.kind != TypeKind.BOOL):
        p.error(f"'_Complex {ty}' is not supported")
    return complex_of(ty)


# ---- small node builders (every call returns a fresh, typed node) ----------

def _vref(p, var, loc):
    return p.typed(N.VarRef(loc, var))


def _part(p, var, which, loc):
    return p.typed(N.Member_(loc, _vref(p, var, loc), var.ty.members[which]))


def _assign(p, loc, lhs, rhs):
    return p.typed(N.Assign(loc, lhs, rhs))


def _seq(p, loc, pre, final):
    """Evaluate `pre` in order, then `final`."""
    node = final
    for e in reversed(pre):
        node = p.typed(N.Comma(loc, e, node))
    return node


def _zero(p, base, loc):
    if is_flonum(base):
        return p.typed(N.FloatLit(loc, 0.0, ty=base))
    return p.typed(N.IntLit(loc, 0, ty=base))


def _check_operand(p, e, loc):
    t = e.ty
    if not (t.is_complex or is_integer(t) or is_flonum(t)):
        p.error(f"invalid operand to complex arithmetic (have '{t}')", loc)


def _to_temp(p, expr, cty, loc):
    """Store `expr` converted to complex type `cty` in a fresh temporary.
    Returns (prelude statements, temporary Var)."""
    pre = []
    if expr.ty.is_complex:
        src = p.new_temp(expr.ty, loc)
        pre.append(_assign(p, loc, _vref(p, src, loc), expr))
        re, im = _part(p, src, 0, loc), _part(p, src, 1, loc)
    else:
        re, im = expr, None
    dst = p.new_temp(complex_of(cty.base), loc)
    pre.append(_assign(p, loc, _part(p, dst, 0, loc), new_cast(re, cty.base)))
    im_val = new_cast(im, cty.base) if im is not None else _zero(p, cty.base, loc)
    pre.append(_assign(p, loc, _part(p, dst, 1, loc), im_val))
    return pre, dst


def _common_base(p, exprs, loc):
    """Usual arithmetic conversion of the parts' types: the widest float if any
    operand is floating, else the common integer type (GCC complex integers)."""
    best = None
    ints = None
    for e in exprs:
        b = e.ty.base if e.ty.is_complex else e.ty
        if b.kind in _RANK:
            if best is None or _RANK[b.kind] > _RANK[best.kind]:
                best = b
        elif is_flonum(b):
            p.error(f"'_Complex' with '{b}' is not supported", loc)
        elif is_integer(b):
            ints = b.unqual() if ints is None else common_type(ints, b).unqual()
    return (best or ints or TY_DOUBLE).unqual()


# ---- public lowering entry points ------------------------------------------

def complex_literal(p, loc, value, cty):
    """Imaginary constant `value * i` of complex type cty."""
    tmp = p.new_temp(cty, loc)
    im = p.typed(N.FloatLit(loc, value, ty=cty.base, bits=getattr(value, "x87", None)))
    pre = [_assign(p, loc, _part(p, tmp, 0, loc), _zero(p, cty.base, loc)),
           _assign(p, loc, _part(p, tmp, 1, loc), im)]
    return CC.attach(_seq(p, loc, pre, _vref(p, tmp, loc)), (0, value), cty.base)


def binary(p, op, lhs, rhs, loc):
    """`lhs op rhs` where at least one operand is complex."""
    _check_operand(p, lhs, loc)
    _check_operand(p, rhs, loc)
    cty = complex_of(_common_base(p, [lhs, rhs], loc))
    pre_a, a = _to_temp(p, lhs, cty, loc)
    pre_b, b = _to_temp(p, rhs, cty, loc)
    pre = pre_a + pre_b

    def A(i):
        return _part(p, a, i, loc)

    def B(i):
        return _part(p, b, i, loc)

    def bin_(o, x, y):
        return new_binop(o, x, y, loc)

    cval = None
    if op in ("==", "!=") or p.fn is None:
        va, vb = CC.value_of(lhs), CC.value_of(rhs)
        if va is not None and vb is not None:
            cval = CC.fold(op, va, vb, cty.base)
    if op in ("==", "!="):
        join = "&&" if op == "==" else "||"
        final = p.typed(N.Logical(loc, join, bin_(op, A(0), B(0)), bin_(op, A(1), B(1))))
        return CC.attach(_seq(p, loc, pre, final), cval, cty.base) if cval else _seq(p, loc, pre, final)
    if op in ("+", "-"):
        re, im = bin_(op, A(0), B(0)), bin_(op, A(1), B(1))
    elif op == "*":
        re = bin_("-", bin_("*", A(0), B(0)), bin_("*", A(1), B(1)))
        im = bin_("+", bin_("*", A(0), B(1)), bin_("*", A(1), B(0)))
    elif op == "/":
        def denom():
            return bin_("+", bin_("*", B(0), B(0)), bin_("*", B(1), B(1)))
        re = bin_("/", bin_("+", bin_("*", A(0), B(0)), bin_("*", A(1), B(1))), denom())
        im = bin_("/", bin_("-", bin_("*", A(1), B(0)), bin_("*", A(0), B(1))), denom())
    else:
        p.error(f"unsupported operator '{op}' on complex operands", loc)
    r = p.new_temp(cty, loc)
    pre += [_assign(p, loc, _part(p, r, 0, loc), re), _assign(p, loc, _part(p, r, 1, loc), im)]
    if op == "/" and cty.base.kind != TypeKind.LDOUBLE:
        # complex long double has its own libgcc helper (__divxc3) and ABI: not lowered
        pre.append(recover_division(p, loc, a, b, r, cty))
    return CC.attach(_seq(p, loc, pre, _vref(p, r, loc)), cval, cty.base)


def negate(p, operand, loc):
    tmp = p.new_temp(operand.ty, loc)
    res = p.new_temp(operand.ty, loc)
    pre = [_assign(p, loc, _vref(p, tmp, loc), operand)]
    for i in (0, 1):
        neg = p.typed(N.Unary(loc, "-", _part(p, tmp, i, loc)))
        pre.append(_assign(p, loc, _part(p, res, i, loc), neg))
    cv = CC.value_of(operand)
    cval = (-cv[0], -cv[1]) if cv is not None else None
    return CC.attach(_seq(p, loc, pre, _vref(p, res, loc)), cval, operand.ty.base)


def conjugate(p, operand, loc):
    """`~z` (GCC extension): the complex conjugate, imaginary part negated."""
    tmp = p.new_temp(operand.ty, loc)
    res = p.new_temp(operand.ty, loc)
    pre = [_assign(p, loc, _vref(p, tmp, loc), operand),
           _assign(p, loc, _part(p, res, 0, loc), _part(p, tmp, 0, loc)),
           _assign(p, loc, _part(p, res, 1, loc),
                   p.typed(N.Unary(loc, "-", _part(p, tmp, 1, loc))))]
    cv = CC.value_of(operand)
    cval = (cv[0], -cv[1]) if cv is not None else None
    return CC.attach(_seq(p, loc, pre, _vref(p, res, loc)), cval, operand.ty.base)


def make_complex(p, loc, re, im):
    """`__builtin_complex(re, im)`: both parts real floating values of one type T
    (the wider one if they differ) give a `_Complex T`."""
    for e in (re, im):
        if e.ty.is_complex or not is_flonum(e.ty):
            p.error("__builtin_complex operands must be real floating-point values", loc)
    base = _common_base(p, [re, im], loc)
    cty = complex_of(base)
    tmp = p.new_temp(cty, loc)
    pre = [_assign(p, loc, _part(p, tmp, 0, loc), new_cast(re, base)),
           _assign(p, loc, _part(p, tmp, 1, loc), new_cast(im, base))]
    vr, vi = CC.real_value(re), CC.real_value(im)
    cval = (vr, vi) if vr is not None and vi is not None else None
    return CC.attach(_seq(p, loc, pre, _vref(p, tmp, loc)), cval, base)


def part(p, operand, which, loc):
    """`__real__ operand` (which=0) or `__imag__ operand` (which=1)."""
    t = operand.ty
    if not t.is_complex:
        if which == 0:
            return operand
        # __imag__ of a real operand is 0, but the operand is still evaluated (PR 33192).
        return _seq(p, loc, [operand], _zero(p, t, loc))
    if isinstance(operand, (N.VarRef, N.Deref, N.Member_)):
        return p.typed(N.Member_(loc, operand, t.members[which]))
    tmp = p.new_temp(t, loc)
    return _seq(p, loc, [_assign(p, loc, _vref(p, tmp, loc), operand)],
                _part(p, tmp, which, loc))


def truth(p, node):
    """A complex value used as a condition: true if either part is nonzero."""
    if not node.ty.is_complex:
        return node
    loc = node.loc
    tmp = p.new_temp(node.ty, loc)
    pre = [_assign(p, loc, _vref(p, tmp, loc), node)]
    base = node.ty.base
    re_nz = new_binop("!=", _part(p, tmp, 0, loc), _zero(p, base, loc), loc)
    im_nz = new_binop("!=", _part(p, tmp, 1, loc), _zero(p, base, loc), loc)
    return _seq(p, loc, pre, p.typed(N.Logical(loc, "||", re_nz, im_nz)))


def convert(p, expr, ty, loc):
    """Implicit or explicit conversion of `expr` to `ty` when either side is complex."""
    if ty.is_complex:
        src = expr.ty
        if src.is_complex and src.base.kind == ty.base.kind and src.base.size == ty.base.size:
            return expr
        _check_operand(p, expr, loc)
        pre, dst = _to_temp(p, expr, ty, loc)
        return CC.attach(_seq(p, loc, pre, _vref(p, dst, loc)), CC.value_of(expr), ty.base)
    if expr.ty.is_complex:
        if not (is_integer(ty) or is_flonum(ty)):
            p.error(f"invalid conversion from '{expr.ty}' to '{ty}'", loc)
        tmp = p.new_temp(expr.ty, loc)
        cv = CC.value_of(expr)
        real = _seq(p, loc, [_assign(p, loc, _vref(p, tmp, loc), expr)],
                    new_cast(_part(p, tmp, 0, loc), ty))
        return CC.attach(real, None if cv is None else (cv[0], 0), ty) if cv is not None else real
    return new_cast(expr, ty)


def builtin_complex(p, loc, args):
    """`__builtin_complex(re, im)`: a complex value from two real floating parts
    of the same type (the parts are not converted, as in GCC)."""
    if len(args) != 2:
        p.error("'__builtin_complex' expects 2 arguments", loc)
    re, im = args
    for e in args:
        if e.ty.is_complex or e.ty.kind not in _RANK:
            p.error("'__builtin_complex' expects real floating-point arguments", loc)
    base = re.ty.unqual()
    if im.ty.kind != base.kind or im.ty.size != base.size:
        p.error("'__builtin_complex' arguments must have the same type", loc)
    tmp = p.new_temp(complex_of(base), loc)
    pre = [_assign(p, loc, _part(p, tmp, 0, loc), re),
           _assign(p, loc, _part(p, tmp, 1, loc), im)]
    node = _seq(p, loc, pre, _vref(p, tmp, loc))
    cre, cim = CC.value_of(re), CC.value_of(im)
    if cre is not None and cim is not None:
        CC.attach(node, (cre[0], cim[0]), base)
    return node
