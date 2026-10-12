"""__builtin_has_attribute(decl, attr): compile-time test whether a declared
function carries a GNU attribute. Only attributes that hcc records on
Function objects are reported; any other name yields 0.

has_attribute_call(parser, tok) -> IntLit   (called with '(' already consumed;
the caller expects the closing ')')
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T

# attribute name -> Function field that records it
FUNCTION_ATTRS = {"noreturn": "is_noreturn", "weak": "is_weak"}


def strip_attr_name(text: str) -> str:
    """`__noreturn__` -> `noreturn`."""
    if text.startswith("__") and text.endswith("__") and len(text) > 4:
        return text[2:-2]
    return text


def has_attribute_call(parser, tok) -> N.Expr:
    decl = parser.typed(parser.assign())
    parser.expect(",")
    name_tok = parser.next()
    field = FUNCTION_ATTRS.get(strip_attr_name(name_tok.text))
    var = getattr(decl, "var", None)
    val = 0
    if field is not None and isinstance(var, N.Function):
        val = int(bool(getattr(var, field)))
    return N.IntLit(tok.loc, val, ty=T.TY_INT)
