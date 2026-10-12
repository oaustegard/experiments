"""GCC's _Float16 math builtins: __builtin_<fn>f16(args) (e.g. __builtin_sinf16).

There is no libm function for binary16, so the call is made to the float
function <fn>f with every floating argument converted to float, and the float
result is converted back to _Float16 (one rounding, as GCC's own expansion).
Integer-returning functions keep their result type. Pointer arguments whose
pointee is _Float16 are rejected: passing them to the float function would
store 4 bytes into a 2-byte object.
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.conversions import new_cast

# functions returning _Float16 (computed by the float version)
F16_FLOAT_RESULT = frozenset("""
acos acosh asin asinh atan atanh atan2 cbrt ceil copysign cos cosh erf erfc exp exp2
expm1 fabs fdim floor fma fmax fmin fmod frexp hypot ldexp lgamma log log10 log1p log2 logb
modf nextafter pow remainder remquo rint round scalbln scalbn sin sinh sqrt tan tanh tgamma trunc
""".split())
# functions returning an integer type
F16_INT_RESULT = {"ilogb": T.TY_INT, "llrint": T.TY_LLONG, "lrint": T.TY_LONG,
                  "llround": T.TY_LLONG, "lround": T.TY_LONG}
# integer parameters (the rest of the floating-point parameters become float)
F16_INT_PARAM = {"ldexp": {1: T.TY_INT}, "scalbn": {1: T.TY_INT}, "scalbln": {1: T.TY_LONG},
                 "frexp": {1: None}, "remquo": {2: None}}


F16_MATH_NAMES = tuple(sorted(F16_FLOAT_RESULT | frozenset(F16_INT_RESULT)))


def _modf_call(parser, tok, args):
    """__builtin_modff16(x, ip): modff on float temporaries; the integral part is
    stored into *ip as _Float16 and the fractional part is the result."""
    loc = tok.loc
    if len(args) != 2 or args[1].ty.kind != T.TypeKind.PTR:
        diag.error(tok, "__builtin_modff16 takes a value and a _Float16 pointer")
    ipart, frac = parser.new_temp(T.TY_FLOAT, loc), parser.new_temp(T.TY_FLOAT, loc)
    fty = T.func_type(T.TY_FLOAT)
    fty.params = [T.TY_FLOAT, T.pointer_to(T.TY_FLOAT)]
    fty.has_proto = True
    fn = parser.implicit_function("modff", tok, T.TY_FLOAT)
    fn.ty = fty
    ptr_f = parser.typed(N.AddrOf(loc, parser.typed(N.VarRef(loc, ipart))))
    call = parser.typed(N.Call(loc, parser.typed(N.VarRef(loc, fn)),
                               [new_cast(args[0], T.TY_FLOAT), ptr_f], fty))
    store_frac = parser.typed(N.Assign(loc, parser.typed(N.VarRef(loc, frac)), call))
    store_int = parser.typed(N.Assign(loc, parser.typed(N.Deref(loc, args[1])),
                                      new_cast(parser.typed(N.VarRef(loc, ipart)), T.TY_FLOAT16)))
    result = new_cast(parser.typed(N.VarRef(loc, frac)), T.TY_FLOAT16)
    rest = parser.typed(N.Comma(loc, store_int, result))
    return parser.typed(N.Comma(loc, store_frac, rest))


def f16_math_call(parser, tok, base: str):
    """Lower `__builtin_<base>f16(...)`; the parser has consumed the name, not '('."""
    loc = tok.loc
    args = parser._args()
    if base == "modf":
        return _modf_call(parser, tok, args)
    int_params = F16_INT_PARAM.get(base, {})
    conv, params = [], []
    for i, a in enumerate(args):
        if a.ty.kind == T.TypeKind.PTR and a.ty.base is not None \
                and a.ty.base.kind == T.TypeKind.FLOAT16:
            diag.error(tok, f"__builtin_{base}f16: _Float16 pointer argument is not supported")
        if i in int_params and int_params[i] is not None:
            pty = int_params[i]
        elif a.ty.kind in T.FLOAT_KINDS or T.is_integer(a.ty) and i not in int_params:
            if base in F16_INT_RESULT and i == 0 or a.ty.kind in T.FLOAT_KINDS or i == 0:
                pty = T.TY_FLOAT
            else:
                pty = a.ty
        else:
            pty = a.ty                  # pointers (frexp/remquo out-parameters)
        conv.append(new_cast(a, pty))
        params.append(pty)
    ret = F16_INT_RESULT.get(base, T.TY_FLOAT)
    fty = T.func_type(ret)
    fty.params = params
    fty.has_proto = True
    fn = parser.implicit_function(base + "f", tok, ret)
    fn.ty = fty
    callee = parser.typed(N.VarRef(loc, fn))
    call = parser.typed(N.Call(loc, callee, conv, fty))
    if ret.kind == T.TypeKind.FLOAT:
        return parser.typed(new_cast(call, T.TY_FLOAT16))
    return call
