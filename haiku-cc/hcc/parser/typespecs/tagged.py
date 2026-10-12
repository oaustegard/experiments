"""struct / union / enum specifiers (parser/structs.py, parser/enums.py)."""
from __future__ import annotations

from hcc import typesys as T
from hcc.parser.typespecs import type_specifier


@type_specifier("struct")
def _struct(self, tok, attr):
    return self.struct_decl(T.TypeKind.STRUCT)


@type_specifier("union")
def _union(self, tok, attr):
    return self.struct_decl(T.TypeKind.UNION)


@type_specifier("enum")
def _enum(self, tok, attr):
    return self.enum_specifier(attr)
