"""__builtin_has_attribute(operand, name): compile-time test whether the operand
carries a GNU attribute.

Two sources are consulted:
  * the operand's *written* attribute list: every name inside
    `__attribute__((...))` that appears outside a brace body (this covers types
    and struct definitions, which are parsed but never evaluated);
  * for an expression naming a function, the Function fields hcc records
    (FUNCTION_ATTRS), e.g. `noreturn`, `weak`.

has_attribute_call(parser, tok) -> IntLit   (called with '(' already consumed;
this function consumes the closing ')' via the caller's expect, as before)
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.parser.attr_copy import member_has_attr

# attribute name -> Function field that records it
FUNCTION_ATTRS = {"noreturn": "is_noreturn", "weak": "is_weak"}


def strip_attr_name(text: str) -> str:
    """`__noreturn__` -> `noreturn`."""
    if text.startswith("__") and text.endswith("__") and len(text) > 4:
        return text[2:-2]
    return text


def written_attribute_names(toks) -> set:
    """Names (leading/trailing underscores stripped) of every attribute written as
    `__attribute__((...))` in `toks`, ignoring anything inside a brace body."""
    names = set()
    braces = 0
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.text == "{":
            braces += 1
        elif t.text == "}":
            braces -= 1
        elif t.text == "__attribute__" and braces == 0:
            depth = 0
            i += 1
            while i < len(toks):
                u = toks[i]
                if u.text == "(":
                    depth += 1
                elif u.text == ")":
                    depth -= 1
                    if depth == 0:
                        break
                elif u.text and (u.text[0].isalpha() or u.text[0] == "_") and depth == 2:
                    names.add(u.text.strip("_"))
                i += 1
        i += 1
    return names


def has_attribute_call(parser, tok) -> N.Expr:
    start = parser.pos
    while parser.at_attribute():         # leading attributes: ATTR((x)) int
        parser.next()
        parser.skip_balanced()
    decl = None
    if parser.is_typename():
        parser.scoped_typename()
    else:
        decl = parser.typed(parser.assign())
    end = parser.pos
    parser.expect(",")
    name = parser.expect_ident().text
    want = strip_attr_name(name).strip("_")
    found = want in written_attribute_names(parser.toks[start:end])
    if isinstance(decl, N.Member_):
        found = found or member_has_attr(decl, want)
    var = getattr(decl, "var", None)
    field = FUNCTION_ATTRS.get(want)
    if field is not None and isinstance(var, N.Function):
        found = found or bool(getattr(var, field))
    return N.IntLit(tok.loc, int(found), ty=T.TY_INT)
