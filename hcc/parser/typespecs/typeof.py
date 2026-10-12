"""typeof / typeof_unqual (and their __typeof__ spellings, via literals.KEYWORD_ALIASES)."""
from __future__ import annotations

from hcc.parser.typespecs import type_specifier


@type_specifier("typeof")
def _typeof(self, tok, attr):
    return self.typeof_specifier()


@type_specifier("typeof_unqual")
def _typeof_unqual(self, tok, attr):
    return self.typeof_specifier(unqual=True)
