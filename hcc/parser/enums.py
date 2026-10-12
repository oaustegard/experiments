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


def fits_type(v, ty):
    """True if integer value v is representable in integer type ty."""
    bits = ty.size * 8
    if ty.is_unsigned:
        return 0 <= v < (1 << bits)
    return -(1 << (bits - 1)) <= v < (1 << (bits - 1))


def enumerator_type(v, expr_ty):
    """C23 type of an explicitly valued enumerator: int if it fits, else the
    type of its constant expression."""
    return T.TY_INT if fits_type(v, T.TY_INT) else expr_ty


def widen_type(ty):
    """Next wider type of the same signedness (implicit enumerator overflow)."""
    if ty.size >= 8:
        return ty
    if ty.is_unsigned:
        return T.TY_UINT if ty.size < 4 else T.TY_ULONG
    return T.TY_INT if ty.size < 4 else T.TY_LONG


class EnumMixin:

    def enum_specifier(self, attr=None):
        self.parse_attributes(VarAttr())
        tag = None
        if self.tok.kind == TokenKind.IDENT:
            tag = self.next()
        self.parse_attributes(VarAttr())
        fixed = self.fixed_underlying()
        if tag is not None and not self.at("{"):
            ty = self.find_tag(tag.text)
            if ty is None:
                ty = T.enum_type()
                ty.tag = tag.text
                self.scopes[-1].tags[tag.text] = ty
            if fixed is not None:
                self.apply_fixed_underlying(ty, fixed)
            return ty
        self.expect("{")
        ty = T.enum_type()
        ty.tag = tag.text if tag else None
        if fixed is not None:
            self.apply_fixed_underlying(ty, fixed)
        if tag is not None:
            # C11 6.2.1p7: the tag's scope begins just after its appearance, so
            # `enum A { X = sizeof(enum A *) }` refers to this (incomplete) type.
            self.scopes[-1].tags[tag.text] = ty
        consts = []
        val = 0
        cty = T.TY_INT
        while not self.accept("}"):
            if consts:
                self.expect(",")
                if self.accept("}"):
                    break
            name = self.expect_ident()
            self.parse_attributes(VarAttr())
            if self.accept("="):
                val, ety = self.const_expr_typed()
                cty = enumerator_type(val, ety)
            elif not fits_type(val, cty):
                cty = widen_type(cty)
            vs = self.push_scope_entry(name.text)
            vs.enum_ty = ty
            vs.enum_val = val
            vs.enum_cty = cty
            consts.append(val)
            val += 1
        lo = min(consts, default=0)
        hi = max(consts, default=0)
        if fixed is not None:
            pass
        elif lo < 0:
            if lo < -(1 << 31) or hi >= 1 << 31:
                ty.size = ty.align = 8
            ty.is_unsigned = False
        elif hi >= 1 << 32:
            ty.size = ty.align = 8
        self.parse_attributes(attr if attr is not None else VarAttr())   # `enum {..} __attribute__((mode(word)))`
        if tag is not None:
            self.scopes[-1].tags[tag.text] = ty
        return ty

    def fixed_underlying(self):
        """C23 `enum e : type`: consume `: type-specifiers` if a type follows.

        A bare `:` (e.g. an unnamed bit-field `enum e : 3;`) is left alone.
        Returns the underlying Type or None.
        """
        if not self.at(":"):
            return None
        i = self.pos + 1
        if i >= len(self.toks) or not self.is_typename(self.toks[i]):
            return None
        self.next()
        return self.declspec()

    def apply_fixed_underlying(self, ty, under):
        ty.size = ty.align = under.size
        ty.is_unsigned = under.is_unsigned
        if under.kind == T.TypeKind.BOOL:
            ty.kind = T.TypeKind.BOOL     # `enum e : _Bool`: conversions to it are `!= 0`
