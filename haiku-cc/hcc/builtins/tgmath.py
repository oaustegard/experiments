"""__builtin_tgmath(f, d, l, ..., args): candidate choice by common floating type
(parser/tgmath.py)."""
from __future__ import annotations

from hcc.builtins import builtin_parser
from hcc.parser.tgmath import tgmath_call


@builtin_parser("__builtin_tgmath")
def _parse_tgmath(self, tok):
    return tgmath_call(self, tok)
