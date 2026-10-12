"""Floating classification: __builtin_isinf/isnan/isfinite[f|l] and finite[f|l],
__builtin_fpclassify and __builtin_isinf_sign (lowered bit-exactly by
codegen/fpclassify.py), __builtin_isnormal (on the libc __fpclassify)."""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.builtins import builtin_parser
from hcc.parser.fp_classify import FP_CLASS_STEMS
from hcc.sema.constexpr import try_eval_const
from hcc.sema.conversions import new_cast


def _fp_value(arg):
    """Floating operands keep their own width (classified bit-exactly);
    integer operands are converted to double like the C library macros do."""
    return arg if T.is_flonum(arg.ty) else new_cast(arg, T.TY_DOUBLE)


@builtin_parser("__builtin_fpclassify")
def _parse_fpclassify(self, tok):
    """__builtin_fpclassify(nan, inf, normal, subnormal, zero, x): the five
    category constants must be integer constant expressions (glibc math.h)."""
    args = self._args(6)
    consts = []
    for a in args[:5]:
        v = try_eval_const(a)
        if v is None:
            self.error("__builtin_fpclassify: category arguments must be integer constants", tok)
        consts.append(N.IntLit(a.loc, v, ty=T.TY_INT))
    return self._call(tok, "fpclassify", consts + [_fp_value(args[5])], T.TY_INT)


@builtin_parser("__builtin_isinf_sign")
def _parse_isinf_sign(self, tok):
    return self._call(tok, "isinf_sign", [_fp_value(self._args(1)[0])], T.TY_INT)


def _fp_class_call(self, tok, base):
    """__builtin_isnan/isinf/isfinite[f|l](x): a 0/1 predicate on x's bit pattern at
    its own width (codegen/fpclassify.py); the f/l spellings convert x first."""
    arg = self._args(1)[0]
    spelled = tok.text[len("__builtin_"):]
    sfx = "" if spelled in FP_CLASS_STEMS else spelled[-1]
    ty = {"f": T.TY_FLOAT, "l": T.TY_LDOUBLE}.get(sfx)
    operand = new_cast(arg, ty) if ty is not None else _fp_value(arg)
    return self._call(tok, base, [operand], T.TY_INT)


@builtin_parser("__builtin_isinf", "__builtin_isinff", "__builtin_isinfl")
def _parse_isinf(self, tok):
    return _fp_class_call(self, tok, "isinf")


@builtin_parser("__builtin_isnan", "__builtin_isnanf", "__builtin_isnanl")
def _parse_isnan(self, tok):
    return _fp_class_call(self, tok, "isnan")


@builtin_parser("__builtin_isfinite", "__builtin_isfinitef", "__builtin_isfinitel",
                "__builtin_finite", "__builtin_finitef", "__builtin_finitel")
def _parse_isfinite(self, tok):
    return _fp_class_call(self, tok, "isfinite")


@builtin_parser("__builtin_isnormal")
def _parse_isnormal(self, tok):
    """__builtin_isnormal(x) == (__fpclassify{,f,l}(x) == FP_NORMAL). glibc has
    no isnormal() symbol, so it is built on the libc fpclassify classifier.
    The operand is evaluated once into a temp. _Float16 has no libc classifier:
    it is widened exactly to double, and a value is normal there iff its
    magnitude is at least 2^-14 (the smallest normal _Float16)."""
    arg = self._args(1)[0]
    f16 = arg.ty.kind == T.TypeKind.FLOAT16
    sfx = {T.TypeKind.FLOAT: "f", T.TypeKind.LDOUBLE: "l"}.get(arg.ty.kind, "")
    cty = {"f": T.TY_FLOAT, "l": T.TY_LDOUBLE}.get(sfx, T.TY_DOUBLE)
    tmp = self.new_temp(cty, tok.loc)
    store = self.typed(N.Assign(tok.loc, self.typed(N.VarRef(tok.loc, tmp)),
                                new_cast(arg, cty)))
    fn = self.implicit_function("__fpclassify" + sfx, tok, T.TY_INT)
    callee = self.typed(N.VarRef(tok.loc, fn))
    val = self.typed(N.VarRef(tok.loc, tmp))
    cls = self.typed(N.Call(tok.loc, callee, [val], fn.ty))
    res = self.typed(N.Binary(tok.loc, "==", cls, N.IntLit(tok.loc, 4, ty=T.TY_INT)))
    if f16:
        lim = 2.0 ** -14
        big = self.typed(N.Logical(tok.loc, "||",
            self.typed(N.Binary(tok.loc, ">=", val, N.FloatLit(tok.loc, lim, ty=T.TY_DOUBLE))),
            self.typed(N.Binary(tok.loc, "<=", self.typed(N.VarRef(tok.loc, tmp)),
                                N.FloatLit(tok.loc, -lim, ty=T.TY_DOUBLE)))))
        res = self.typed(N.Logical(tok.loc, "&&", res, big))
    return self.typed(N.Comma(tok.loc, store, res))
