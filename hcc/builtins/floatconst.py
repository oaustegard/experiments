"""Floating constants: __builtin_{inf,huge_val,nan,nans}{,f,l}, their GCC _FloatN
spellings (__builtin_inff64x, ...) and the _Float16 forms (__builtin_inff16 and
the f16 math functions of parser/float16_math.py)."""
from __future__ import annotations

import math

from hcc import nodes as N
from hcc import typesys as T
from hcc.builtins import builtin_parser
from hcc.halfbits import half_to_double
from hcc.parser.float16_math import F16_MATH_NAMES, f16_math_call
from hcc.parser.nan_payload import nan_literal

# GCC's _FloatN spellings of the inf/nan builtins, e.g. __builtin_inff64x (the
# _Float64x variant). Mapped onto the base builtin of the matching C type width:
# _Float32 -> float, _Float32x/_Float64 -> double, _Float64x/_Float128 -> long double.
_FLOATN_SUFFIX = {"32": "f", "32x": "", "64": "", "64x": "l", "128": "l", "128x": "l"}
_SUFFIX_TY = {"": T.TY_DOUBLE, "f": T.TY_FLOAT, "l": T.TY_LDOUBLE}


def floatn_libc_name(name: str) -> str:
    """`fabsf64x` -> `fabsl`, `acoshf32` -> `acoshf`: the libm function behind a GCC
    __builtin_<fn>f<N>[x] spelling (the libm suffix of the same C type width).
    Other names are returned unchanged."""
    for width, sfx in _FLOATN_SUFFIX.items():
        tail = "f" + width
        if name.endswith(tail) and len(name) > len(tail):
            return name[:-len(tail)] + sfx
    return name


def _spellings(base: str, sfx: str) -> list:
    """__builtin_<base><sfx> plus every _FloatN spelling of the same C type."""
    return [f"__builtin_{base}{sfx}"] + [f"__builtin_{base}f{w}" for w, s in _FLOATN_SUFFIX.items() if s == sfx]


def _inf_lit(self, tok, ty):
    """__builtin_huge_val*()/__builtin_inf*(): a constant +infinity."""
    return N.FloatLit(tok.loc, float("inf"), ty=ty)


def _nan_lit(self, tok, ty):
    """__builtin_nan*(tag): a constant quiet NaN whose payload is read from the tag."""
    args = self._args(1)
    return nan_literal(args[0], ty, False, tok.loc)


def _snan_lit(self, tok, ty):
    """__builtin_nans*(tag): a constant signaling NaN (quiet bit clear)."""
    args = self._args(1)
    return nan_literal(args[0], ty, True, tok.loc)


def _register(base: str, sfx: str, lit) -> None:
    ty = _SUFFIX_TY[sfx]

    def parse_const(self, tok):
        return lit(self, tok, ty)
    parse_const.__qualname__ = f"parse_{base}{sfx}"
    builtin_parser(*_spellings(base, sfx))(parse_const)


for _sfx in ("", "f", "l"):
    _register("inf", _sfx, _inf_lit)
    _register("huge_val", _sfx, _inf_lit)
    _register("nan", _sfx, _nan_lit)
    _register("nans", _sfx, _snan_lit)


@builtin_parser(*(f"__builtin_{base}f16" for base in ("inf", "huge_val", "nan", "nans") + F16_MATH_NAMES))
def _parse_float16(self, tok):
    """__builtin_{inf,huge_val,nan,nans}f16(...): a _Float16 constant. The NaN bit
    patterns are built directly in binary16 (a float or double round trip would
    quiet a signaling NaN). Other math functions go through parser/float16_math.py."""
    base = tok.text[len("__builtin_"):-len("f16")]
    if base in F16_MATH_NAMES:
        return f16_math_call(self, tok, base)
    if base.startswith("nan"):
        self._args(1)
        val = half_to_double(0x7D01 if base == "nans" else 0x7E00)
    else:
        self._args(0)
        val = math.inf
    return N.FloatLit(tok.loc, val, ty=T.TY_FLOAT16)
