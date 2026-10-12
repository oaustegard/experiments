"""Parser package.

    parse(tokens, opts=None) -> nodes.TranslationUnit

`Parser` is assembled from one mixin per grammar area (see CONTRACT section 2).
All mixins share state through ParserBase (parser/core.py). Each Expr node is
typed immediately via self.typed() -> sema.typing.add_type.
"""
from __future__ import annotations

from hcc.parser.attributes import AttributeMixin
from hcc.parser.builtins import BuiltinParseMixin
from hcc.parser.core import ParserBase
from hcc.parser.declarator import DeclaratorMixin
from hcc.parser.declspec import DeclSpecMixin
from hcc.parser.enums import EnumMixin
from hcc.parser.expressions import ExpressionMixin
from hcc.parser.initializers import InitializerMixin
from hcc.parser.locals import LocalDeclMixin
from hcc.parser.nested_fn import NestedFnMixin
from hcc.parser.postfix import PostfixMixin
from hcc.parser.primary import PrimaryMixin
from hcc.parser.statements import StatementMixin
from hcc.parser.structs import StructMixin
from hcc.parser.toplevel import TopLevelMixin
from hcc.parser.vla import VlaMixin
from hcc.parser.va_pack import specialise_pack_calls


class Parser(TopLevelMixin, LocalDeclMixin, NestedFnMixin, StatementMixin, InitializerMixin,
             ExpressionMixin, PostfixMixin, PrimaryMixin, BuiltinParseMixin,
             DeclSpecMixin, DeclaratorMixin, StructMixin, EnumMixin, AttributeMixin,
             VlaMixin, ParserBase):
    pass


def parse(tokens, opts=None):
    tu = Parser(tokens, opts).translation_unit()
    specialise_pack_calls(tu)
    return tu
