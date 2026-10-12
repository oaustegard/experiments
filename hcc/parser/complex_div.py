"""C99 Annex G recovery for complex division (GCC's default complex_method).

complex.binary() computes `a / b` with the textbook (Smith-free) formula into a
temporary r.  When the quotient comes out NaN+iNaN although the operands were
not NaN (e.g. a nonzero numerator over a zero denominator, or an infinite
operand), C requires the infinities / zeros of Annex G instead.  Like GCC, we
then call libgcc's helper `__divdc3` (or `__divsc3` for float), which
implements those rules:

    if (r.re != r.re && r.im != r.im) r = __divdc3(a.re, a.im, b.re, b.im);

The helper is an ordinary external function (libgcc is always linked by the
gcc driver), so the struct-return ABI of `_Complex` applies unchanged.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.parser.scope import VarScope
from hcc.sema.arith import new_binop
from hcc.typesys import TypeKind, func_type


def _helper(p, cty, loc):
    """The libgcc division helper for complex type cty, declared once with a prototype."""
    name = "__divsc3" if cty.base.kind == TypeKind.FLOAT else "__divdc3"
    vs = p.find_var(name)
    if vs is not None and vs.var is not None:
        return vs.var
    fty = func_type(cty)
    fty.params = [cty.base] * 4
    fty.has_proto = True
    fn = N.Function(name, fty, loc=loc)
    p.scopes[0].vars.setdefault(name, VarScope()).var = fn
    p.objs.append(fn)
    return fn


def recover_division(p, loc, a, b, r, cty):
    """Complex-valued conditional: if r is NaN+iNaN, set r = helper(a, b).

    a, b, r are complex temporaries (VarRef-able Vars) of type cty; r already
    holds the inline quotient.  Every call builds fresh nodes."""
    def part(var, i):
        return p.typed(N.Member_(loc, p.typed(N.VarRef(loc, var)), cty.members[i]))

    def is_nan(i):
        x = part(r, i)
        return new_binop("!=", x, part(r, i), loc)

    test = p.typed(N.Logical(loc, "&&", is_nan(0), is_nan(1)))
    fn = p.typed(N.VarRef(loc, _helper(p, cty, loc)))
    call = p.build_call(fn, [part(a, 0), part(a, 1), part(b, 0), part(b, 1)], loc)
    then = p.typed(N.Assign(loc, p.typed(N.VarRef(loc, r)), call))
    # the else arm is r itself (a complex value), so both arms have the same type
    return p.typed(N.Cond(loc, test, then, p.typed(N.VarRef(loc, r))))
