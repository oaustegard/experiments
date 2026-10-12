"""Parser package.

    parse(tokens, opts=None) -> nodes.TranslationUnit

`Parser` is assembled from one mixin per grammar area (see CONTRACT section 2):
every `*Mixin` class defined in a module of this package, found by hcc.registry.
All mixins share state through ParserBase (parser/core.py). Each Expr node is
typed immediately via self.typed() -> sema.typing.add_type.
"""
from __future__ import annotations

from hcc.parser.core import ParserBase
from hcc.parser.va_pack import specialise_pack_calls
from hcc.registry import assemble_mixins


# Every `*Mixin` class in a module of hcc/parser/ is a base (hcc/registry.py):
# a new grammar area is a new file, with no import or base list to edit here.
class Parser(*assemble_mixins("hcc.parser", ParserBase)):
    pass


def parse(tokens, opts=None):
    tu = Parser(tokens, opts).translation_unit()
    specialise_pack_calls(tu)
    return tu
