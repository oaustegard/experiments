"""Type-specifier keywords that build a whole type (struct, enum, typeof, ...).

One module per keyword family, each handler registered with
`@type_specifier("<keyword>")`; parser/declspec.py dispatches on the registry
and counts the keyword as starting a declaration (is_typename), so a new one
needs no edit there:

    @type_specifier("__foo")          # hcc/parser/typespecs/foo.py
    def _foo(self, tok, attr):        # self = Parser; the keyword is already consumed
        return T.TY_INT               # the Type (None: inferred, as for __auto_type)

The spelling must be a keyword token (tokens.KEYWORDS, or a literals.py alias).
"""
from __future__ import annotations

from hcc.registry import Registry

type_specifier = Registry("type specifier", "hcc.parser.typespecs")
