"""struct / union declarations and layout.

StructMixin provides:
    struct_decl(kind) -> Type          after the `struct`/`union` keyword
    find_member(ty, name) -> list[Member] | None
        path of members (several when reaching into anonymous members)
Layout (offsets, alignment, bitfields, packed, flexible array) is in
layout_struct()/layout_union(), following the SysV/gcc rules.
"""
from __future__ import annotations

from hcc import diag
from hcc.parser.scope import VarAttr
from hcc.parser.pragma_pack import at_pack_marker, consume_pack_marker
from hcc.tokens import TokenKind
from hcc import typesys as T


def _kind_word(kind) -> str:
    return "struct" if kind == T.TypeKind.STRUCT else "union"


class StructMixin:

    def struct_decl(self, kind):
        attr = VarAttr()
        self.parse_attributes(attr)
        tag = None
        if self.tok.kind == TokenKind.IDENT:
            tag = self.next()
        self.parse_attributes(attr)
        if tag is not None and not self.at("{"):
            ty = self.find_tag(tag.text)
            if ty is not None and not (self.at(";") and self.tag_is_shadow_decl(tag.text)):
                return ty
            ty = T.struct_type(kind)
            ty.tag = tag.text
            self.scopes[-1].tags[tag.text] = ty
            return ty
        brace_pos = self.pos
        def_pos = self.__dict__.setdefault("struct_def_pos", {})   # id(ty) -> '{' index
        self.expect("{")
        ty = None
        if tag is not None:
            prev = self.scopes[-1].tags.get(tag.text)
            if prev is not None and prev.is_complete and prev.kind == kind:
                kw = "struct" if kind == T.TypeKind.STRUCT else "union"
                self.error(f"redefinition of '{kw} {tag.text}'", tag)
            if prev is not None and not prev.is_complete and prev.kind == kind:
                ty = prev
            elif prev is not None and prev.is_complete and prev.kind == kind:
                # a second body for the same tag in one scope; a re-parse of the
                # same declaration (compound-literal backtrack) is not a redefinition
                # (types stay alive in the scopes that name them, so id() is stable)
                if def_pos.get(id(prev)) != brace_pos:
                    diag.error(tag, f"redefinition of '{_kind_word(kind)} {tag.text}'")
        if ty is None:
            ty = T.struct_type(kind)
            ty.tag = tag.text if tag else None
            if tag is not None:
                self.scopes[-1].tags[tag.text] = ty
                def_pos[id(ty)] = brace_pos
        members = self.struct_members()
        self.parse_attributes(attr)
        ty.members = members
        ty.is_packed = attr.is_packed
        ty.is_ms_struct = attr.is_ms_struct
        ty.is_reverse_sso = attr.reverse_sso and kind == T.TypeKind.STRUCT
        if kind == T.TypeKind.STRUCT:
            layout_struct(ty)
        else:
            layout_union(ty)
        if attr.align:
            ty.align = max(ty.align, attr.align)
            ty.size = T.align_to(ty.size, ty.align)
        if ty.is_reverse_sso:
            mirror_reverse_bitfields(ty)
            mark_reverse_scalars(ty)
        ty.is_complete = True
        return ty

    def tag_is_shadow_decl(self, name: str) -> bool:
        """`struct S;` in an inner scope declares a new incomplete type."""
        return name not in self.scopes[-1].tags

    def struct_members(self) -> list:
        members = []
        while not self.accept("}"):
            if at_pack_marker(self):        # #pragma pack inside the body applies to later members
                consume_pack_marker(self)
                continue
            if self.at("_Static_assert"):
                self.static_assert()
                continue
            if self.accept(";"):
                continue
            self.skip_std_attributes()     # `[[gnu::x]] char c[];`
            attr = VarAttr()
            start = self.pos
            basety = self.declspec(attr)
            single_tok = self.pos == start + 1      # the declspec was one bare identifier
            # GNU: a struct/union specifier may end a member list without its `;`
            # (`struct { struct { int a; } }`), which is treated as an anonymous member.
            missing_semi = self.at("}") and basety.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION)
            if self.accept(";") or missing_semi:
                # anonymous struct/union member; `struct S;` (tagged) declares nothing
                # unless -fplan9-extensions makes it an anonymous member of type S
                plan9 = self.opts is not None and self.opts.plan9_extensions
                if basety.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION) and (basety.tag is None or plan9):
                    m = T.Member(None, basety, 0, attr.align or basety.align)
                    m.pack_cap = self.pack_align
                    tok0 = self.toks[start]
                    if plan9 and single_tok and tok0.kind == TokenKind.IDENT:
                        m.alias = tok0.text     # `typedef struct {..} D; struct { D; };` names it D
                    members.append(m)
                continue
            first = True
            while not self.accept(";"):
                if not first and self.at("}"):
                    # GNU: `struct { int a }` — the last member lacks ';' before '}'
                    diag.warn(self.toks[self.pos], "no semicolon at end of struct or union")
                    break
                if not first:
                    self.expect(",")
                first = False
                mattr = VarAttr()
                ty, name = self.declarator(basety)
                self.parse_attributes(mattr)
                check_member_complete(ty, name, self.toks[start])
                m = T.Member(name.text if name else None, ty)
                m.tok = name
                m.pack_cap = self.pack_align
                # aligned/_Alignas on a member can only raise its alignment
                m.explicit_align = max(mattr.align, attr.align)
                m.align = max(ty.align, m.explicit_align)
                if attr.is_packed:      # leading `__attribute__((packed | copy(..)))` on the member
                    m.align = m.explicit_align or 1     # packed keeps an explicit aligned(N)
                    m.is_packed = True
                if self.accept(":"):
                    m.is_bitfield = True
                    m.bit_width = self.const_expr()
                self.parse_attributes(mattr)
                m.explicit_align = max(m.explicit_align, mattr.align)
                if m.explicit_align:
                    m.align = max(m.align, m.explicit_align)
                if mattr.is_packed:
                    m.align = 1
                    m.is_packed = True
                members.append(m)
        if members and members[-1].ty.kind == T.TypeKind.ARRAY and members[-1].ty.array_len < 0:
            last = members[-1]
            last.ty = T.array_of(last.ty.base, 0)
        for i, m in enumerate(members):
            m.idx = i
        return members


def check_member_complete(ty, name, where) -> None:
    """A member's type (after array dimensions) must be a complete struct/union.
    Pointers to incomplete types and flexible array members (`T a[]`) are fine."""
    base = ty
    while base is not None and base.kind == T.TypeKind.ARRAY:
        base = base.base
    if base is None or base.kind not in (T.TypeKind.STRUCT, T.TypeKind.UNION):
        return
    if complete_struct(base).is_complete:
        return
    if name is not None:
        diag.error(name, f"field '{name.text}' has incomplete type")
    diag.error(where, "field has incomplete type")


def _unnamed_bitfield(m) -> bool:
    """Unnamed bit-fields never raise the alignment of their aggregate (SysV)."""
    return m.is_bitfield and m.name is None


def layout_struct(ty) -> None:
    """Each member's `pack_cap` is the #pragma pack maximum alignment in effect where it was declared."""
    bits = 0
    align = 1
    packed = ty.is_packed
    ms = ty.is_ms_struct
    unit = 0        # ms: bits of the last bit-field's declared-type storage unit
    zero_width_prev = False
    if members_flexible(ty):
        ty.is_flexible = True
    for m in ty.members:
        if packed:      # packed drops natural alignment; explicit aligned(N) still applies
            m.align = 1 if m.is_packed else (m.explicit_align or 1)
        if m.pack_cap and not m.is_bitfield and not m.explicit_align and m.align > m.pack_cap:
            m.align = m.pack_cap    # #pragma pack caps natural member alignment
        if m.is_bitfield and m.bit_width == 0:
            if ms and zero_width_prev:      # MS: back-to-back zero-width bit-fields do not pad again
                continue
            if ms and align < m.ty.size:    # MS: a padding zero-width bit-field aligns the struct
                align = m.ty.size
            bits = T.align_to(bits, m.ty.size * 8)
            unit = 0
            zero_width_prev = True
            continue
        zero_width_prev = False
        if m.is_bitfield:
            sz = m.ty.size
            loose = packed or m.is_packed
            if ms and not unit and bits:    # MS: a bit-field never shares a unit with plain data
                bits = T.align_to(bits, sz * 8)
            if not loose and bits // (sz * 8) != (bits + m.bit_width - 1) // (sz * 8):
                bits = T.align_to(bits, sz * 8)
            m.offset = (bits // 8) // sz * sz if not loose else bits // 8
            m.bit_offset = bits - m.offset * 8
            bits += m.bit_width
            unit = sz * 8
        else:
            if m.ty.kind == T.TypeKind.VLA or T.is_dyn_struct(m.ty):
                _layout_dyn_tail(ty, m, bits, align)
                return
            if ms and unit:     # MS: a following plain member starts after the bit-field's unit
                bits = T.align_to(bits, unit)
            unit = 0
            bits = T.align_to(bits, m.align * 8)
            m.offset = bits // 8
            bits += m.ty.size * 8
        if align < m.align and not _unnamed_bitfield(m):
            align = m.align
    ty.align = align
    ty.size = T.align_to(T.align_to(bits, 8) // 8, align)


def _layout_dyn_tail(ty, m, bits: int, align: int) -> None:
    """Trailing variably-sized member (GNU `struct { int x[n]; }`): the static
    size is the offset of the member; the run-time size is added by sizeof."""
    if m is not ty.members[-1]:
        diag.error(m.tok, "variably modified member must be the last member of a struct", "parse")
    if T.is_dyn_struct(m.ty):
        diag.error(m.tok, "unsupported: struct with variably modified member nested in a struct", "parse")
    if ty.is_packed or ty.is_ms_struct or ty.is_reverse_sso:
        diag.error(m.tok, "unsupported: variably modified member in packed or reversed struct", "parse")
    elem = m.ty.base
    m.align = max(m.explicit_align, elem.align)     # a VLA's own align is its slot size
    off = T.align_to(bits // 8, m.align)
    m.offset = off
    ty.dyn_tail = m.ty
    ty.size = off
    ty.align = max(align, m.align)


def mirror_reverse_bitfields(ty) -> None:
    """scalar_storage_order("big-endian"): the struct image is the byte-reversed
    little-endian image of a bit-mirrored layout (bit p..p+w-1 -> 8S-(p+w)..).
    Each bitfield becomes a big-endian byte span: offset = first memory byte,
    bit_offset = LSB position inside the span, rev_bytes = span length."""
    total = ty.size * 8
    for m in ty.members:
        if not m.is_bitfield or m.bit_width == 0:
            continue
        p = m.offset * 8 + m.bit_offset
        q = total - (p + m.bit_width)
        j0 = q // 8
        j1 = (q + m.bit_width - 1) // 8
        m.offset = ty.size - 1 - j1
        m.bit_offset = q - 8 * j0
        m.rev_bytes = j1 - j0 + 1


def mark_reverse_scalars(ty) -> None:
    """scalar_storage_order("big-endian"): integer and floating members are
    stored byte-reversed. The member type becomes a reverse-order variant
    (Type.is_rev_order) so loads, stores and derefs of it swap bytes; arrays
    get a reverse-order element type. Pointers stay in native order (as gcc
    does). Scalars wider than 8 bytes and long double are rejected instead of
    miscompiled."""
    for m in ty.members:
        if not m.is_bitfield:
            m.ty = _reverse_type(m.ty, m.tok)


def _reverse_type(ty, tok):
    k = ty.kind
    if k in (T.TypeKind.STRUCT, T.TypeKind.UNION):
        return ty                       # nested aggregate keeps its own order
    if k == T.TypeKind.ARRAY:
        if ty.base is None:
            return ty
        base = _reverse_type(ty.base, tok)
        if base is ty.base:
            return ty
        arr = ty.copy()
        arr.base = base
        return arr
    if ty.size == 1 or ty.is_rev_order or k == T.TypeKind.PTR:
        return ty                       # byte-sized or pointer: same image either way
    if k in (T.TypeKind.LDOUBLE, T.TypeKind.FLOAT128, T.TypeKind.VLA, T.TypeKind.INT128) \
            or ty.size > 8:
        diag.error(tok, "scalar_storage_order: unsupported member type in "
                   "reverse-order struct", "parse")
    rev = ty.copy()
    rev.is_rev_order = True
    return rev


def layout_union(ty) -> None:
    size = 0
    align = 1
    for m in ty.members:
        if ty.is_packed:
            m.align = 1
        m.offset = 0
        size = max(size, m.ty.size if not m.is_bitfield else (m.bit_width + 7) // 8)
        if not _unnamed_bitfield(m):
            align = max(align, m.align)
    ty.align = align
    ty.size = T.align_to(size, align)


def members_flexible(ty) -> bool:
    ms = ty.members
    return bool(ms) and ms[-1].ty.kind == T.TypeKind.ARRAY and ms[-1].ty.array_len == 0


def complete_struct(ty):
    """Return the completed type behind a qualified copy made while incomplete.

    `const struct S` copies S's Type before S's body is seen; the copy keeps
    an empty member list, so member lookup follows `origin` to the real type.
    """
    while not ty.is_complete and ty.origin is not None:
        ty = ty.origin
    return ty


def find_member(ty, name: str):
    """Return the member path to `name` (through anonymous members) or None."""
    ty = complete_struct(ty)
    for m in ty.members:
        if m.name is None and m.alias == name:
            return [m]
        if m.name is None and m.ty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION):
            sub = find_member(m.ty, name)
            if sub is not None:
                return [m] + sub
        elif m.name == name:
            return [m]
    return None
