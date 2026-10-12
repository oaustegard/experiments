"""__builtin_complex(re, im) and __builtin_conj{f,,l}(z): built by parser/complex.py
(complex values are STRUCT{re,im} temporaries)."""
from __future__ import annotations

from hcc.builtins import builtin_parser
from hcc.parser.complex import builtin_complex, conjugate


@builtin_parser("__builtin_complex")
def _parse_complex(self, tok):
    return builtin_complex(self, tok.loc, self._args())


@builtin_parser("__builtin_conj", "__builtin_conjf", "__builtin_conjl")
def _parse_conj(self, tok):
    """__builtin_conj{f,,l}(z): the complex conjugate of a _Complex value."""
    (z,) = self._args(1)
    if not z.ty.is_complex:
        self.error("__builtin_conj expects a complex argument")
    return conjugate(self, z, tok.loc)
