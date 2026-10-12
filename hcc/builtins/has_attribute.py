"""__builtin_has_attribute(decl-or-type, attribute): a constant (parser/has_attribute.py)."""
from __future__ import annotations

from hcc.builtins import builtin_parser
from hcc.parser.has_attribute import has_attribute_call


@builtin_parser("__builtin_has_attribute")
def _parse_has_attribute(self, tok):
    return has_attribute_call(self, tok)
