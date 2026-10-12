"""enum specifiers.

EnumMixin.enum_specifier() -> Type   (after the `enum` keyword)
Enumerators are pushed into the current scope as VarScope(enum_ty, enum_val).
Underlying type (gcc): unsigned int if no negative values, else int; widened
to (unsigned) long when values don't fit in 32 bits.
"""
from __future__ import annotations

from hcc.parser.scope import VarAttr
from hcc.tokens import TokenKind
from hcc import typesys as T


class EnumMixin:

    def enum_specifier(self):
        self.parse_attributes(VarAttr())
        tag = None
        if self.tok.kind == TokenKind.IDENT:
            tag = self.next()
        self.parse_attributes(VarAttr())
        if tag is not None and not self.at("{"):
            ty = self.find_tag(tag.text)
            if ty is None:
                ty = T.enum_type()
                ty.tag = tag.text
                self.scopes[-1].tags[tag.text] = ty
            return ty
        self.expect("{")
        ty = T.enum_type()
        ty.tag = tag.text if tag else None
        consts = []
        val = 0
        while not self.accept("}"):
            if consts:
                self.expect(",")
                if self.accept("}"):
                    break
            name = self.expect_ident()
            self.parse_attributes(VarAttr())
            if self.accept("="):
                val = self.const_expr()
            vs = self.push_scope_entry(name.text)
            vs.enum_ty = ty
            vs.enum_val = val
            consts.append(val)
            val += 1
        lo = min(consts, default=0)
        hi = max(consts, default=0)
        if lo < 0:
            if lo < -(1 << 31) or hi >= 1 << 31:
                ty.size = ty.align = 8
            ty.is_unsigned = False
        elif hi >= 1 << 32:
            ty.size = ty.align = 8
        self.parse_attributes(VarAttr())
        if tag is not None:
            self.scopes[-1].tags[tag.text] = ty
        return ty
