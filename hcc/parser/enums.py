"""enum specifiers.

EnumMixin.enum_specifier() -> Type   (after the `enum` keyword)
Enumerators are pushed into the current scope as VarScope(enum_ty, enum_val).
Underlying type (gcc): unsigned int if no negative values, else int; widened
to (unsigned) long when values don't fit in 32 bits.
"""
from __future__ import annotations

from hcc.parser.scope import VarAttr
from hcc.parser.redecl import check_redecl
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
        if tag is not None and fixed is not None:
            self.check_fwd_fixed(tag)
        if tag is not None and not self.at("{"):
            ty = self.find_tag(tag.text)
            if ty is None:
                ty = T.enum_type()
                ty.tag = tag.text
                self.scopes[-1].tags[tag.text] = ty
                if fixed is None:
                    ty.enum_consts = "fwd"    # forward ref: no body, no fixed type (incomplete)
            if fixed is not None:
                self.check_fixed_after_nofixed_body(ty, tag)
                self.apply_fixed_underlying(ty, fixed)
            return ty
        brace_pos = self.pos
        def_pos = self.__dict__.setdefault("enum_def_pos", {})   # id(ty) -> '{' index
        self.expect("{")
        open_defs = self.__dict__.setdefault("enum_open_defs", set())
        if tag is not None:
            prev = self.find_tag(tag.text)     # visible from enclosing scopes too (sizeof(...) opens one)
            if prev is not None and id(prev) in open_defs:
                self.error(f"nested redefinition of 'enum {tag.text}'", tag)
            # A second body for the same tag in one scope is a redeclaration.
            # A re-parse of the same declaration (backtrack) is not.
            prev = self.scopes[-1].tags.get(tag.text)
            if (prev is not None and prev.kind == T.TypeKind.ENUM and id(prev) in def_pos
                    and def_pos[id(prev)] != brace_pos):
                self.error(f"redeclaration of 'enum {tag.text}'", tag)
        prev_fwd = self.scopes[-1].tags.get(tag.text) if tag is not None else None
        if prev_fwd is not None and prev_fwd.kind == T.TypeKind.ENUM and prev_fwd.enum_consts == "fwd":
            ty = prev_fwd                 # the body completes an earlier `enum e;` in this scope
            ty.enum_consts = None
        else:
            ty = T.enum_type()
        ty.tag = tag.text if tag else None
        if tag is not None:
            def_pos[id(ty)] = brace_pos
            # The tag is visible from its declaration on, so enumerator
            # expressions may refer to it (GNU: `enum A { B = sizeof(enum A*) }`).
            self.scopes[-1].tags[tag.text] = ty
        open_defs.add(id(ty))
        try:
            return self.enum_body(ty, tag, fixed, attr)
        finally:
            open_defs.discard(id(ty))

    def enum_body(self, ty, tag, fixed, attr):
        """Enumerator list after the `{` of an enum definition (see enum_specifier)."""
        if fixed is not None:
            self.apply_fixed_underlying(ty, fixed)
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
            check_redecl(self, self.scopes[-1], name, "enum")
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
        elif tag is not None:
            self.__dict__.setdefault("enum_nofixed_ids", {})[id(ty)] = ty   # keeps ty alive: no id reuse
        elif lo < 0:
            if lo < -(1 << 31) or hi >= 1 << 31:
                ty.size = ty.align = 8
            ty.is_unsigned = False
        elif hi >= 1 << 32:
            ty.size = ty.align = 8
        self.parse_attributes(attr if attr is not None else VarAttr())   # `enum {..} __attribute__((mode(word)))`
        self.check_designated_init(attr, False, self.toks[self.pos - 1])
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

    def check_fwd_fixed(self, tag):
        """`enum e;` (no fixed type) followed by `enum e : T` in the same scope is
        an error: the enum was declared both with and without a fixed type."""
        prev = self.scopes[-1].tags.get(tag.text)
        if prev is not None and prev.kind == T.TypeKind.ENUM and prev.enum_consts == "fwd":
            self.error(f"'enum' declared both with and without fixed underlying type", tag)

    def check_fixed_after_nofixed_body(self, ty, tag):
        """`enum e : T` naming a tag whose body (in this scope) was defined without a
        fixed underlying type is an error (e.g. `enum e : typeof(enum e {A})`)."""
        nofixed = self.__dict__.get("enum_nofixed_ids", {})
        if nofixed.get(id(ty)) is ty:
            self.error(f"'enum' declared with but defined without fixed underlying type", tag)

    def apply_fixed_underlying(self, ty, under):
        ty.size = ty.align = under.size
        ty.is_unsigned = under.is_unsigned
        if under.kind == T.TypeKind.BOOL:
            ty.kind = T.TypeKind.BOOL     # `enum e : _Bool`: conversions to it are `!= 0`
