"""struct / union declarations and layout.

StructMixin provides:
    struct_decl(kind) -> Type          after the `struct`/`union` keyword
    find_member(ty, name) -> list[Member] | None
        path of members (several when reaching into anonymous members)
Layout (offsets, alignment, bitfields, packed, flexible array) is in
layout_struct()/layout_union(), following the SysV/gcc rules.
"""
from __future__ import annotations

from hcc.parser.scope import VarAttr
from hcc.tokens import TokenKind
from hcc import typesys as T


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
        self.expect("{")
        ty = None
        if tag is not None:
            prev = self.scopes[-1].tags.get(tag.text)
            if prev is not None and not prev.is_complete and prev.kind == kind:
                ty = prev
        if ty is None:
            ty = T.struct_type(kind)
            ty.tag = tag.text if tag else None
            if tag is not None:
                self.scopes[-1].tags[tag.text] = ty
        members = self.struct_members()
        self.parse_attributes(attr)
        ty.members = members
        ty.is_packed = attr.is_packed
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
        ty.is_complete = True
        return ty

    def tag_is_shadow_decl(self, name: str) -> bool:
        """`struct S;` in an inner scope declares a new incomplete type."""
        return name not in self.scopes[-1].tags

    def struct_members(self) -> list:
        members = []
        while not self.accept("}"):
            if self.at("_Static_assert"):
                self.static_assert()
                continue
            if self.accept(";"):
                continue
            self.skip_std_attributes()     # `[[gnu::x]] char c[];`
            attr = VarAttr()
            basety = self.declspec(attr)
            if self.accept(";"):           # anonymous struct/union member
                m = T.Member(None, basety, 0, attr.align or basety.align)
                members.append(m)
                continue
            first = True
            while not self.accept(";"):
                if not first:
                    self.expect(",")
                first = False
                mattr = VarAttr()
                ty, name = self.declarator(basety)
                self.parse_attributes(mattr)
                m = T.Member(name.text if name else None, ty)
                m.tok = name
                # aligned/_Alignas on a member can only raise its alignment
                m.align = max(ty.align, mattr.align, attr.align)
                if attr.is_packed or mattr.is_packed:
                    m.align = 1
                if self.accept(":"):
                    m.is_bitfield = True
                    m.bit_width = self.const_expr()
                self.parse_attributes(mattr)
                members.append(m)
        if members and members[-1].ty.kind == T.TypeKind.ARRAY and members[-1].ty.array_len < 0:
            last = members[-1]
            last.ty = T.array_of(last.ty.base, 0)
        for i, m in enumerate(members):
            m.idx = i
        return members


def layout_struct(ty) -> None:
    bits = 0
    align = 1
    packed = ty.is_packed
    if members_flexible(ty):
        ty.is_flexible = True
    for m in ty.members:
        if packed:
            m.align = 1
        if m.is_bitfield and m.bit_width == 0:
            bits = T.align_to(bits, m.ty.size * 8)
            continue
        if m.is_bitfield:
            sz = m.ty.size
            if not packed and bits // (sz * 8) != (bits + m.bit_width - 1) // (sz * 8):
                bits = T.align_to(bits, sz * 8)
            m.offset = (bits // 8) // sz * sz if not packed else bits // 8
            m.bit_offset = bits - m.offset * 8
            bits += m.bit_width
        else:
            bits = T.align_to(bits, m.align * 8)
            m.offset = bits // 8
            bits += m.ty.size * 8
        if align < m.align:
            align = m.align
    ty.align = align
    ty.size = T.align_to(T.align_to(bits, 8) // 8, align)


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


def layout_union(ty) -> None:
    size = 0
    align = 1
    for m in ty.members:
        if ty.is_packed:
            m.align = 1
        m.offset = 0
        size = max(size, m.ty.size if not m.is_bitfield else (m.bit_width + 7) // 8)
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
        if m.name is None and m.ty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION):
            sub = find_member(m.ty, name)
            if sub is not None:
                return [m] + sub
        elif m.name == name:
            return [m]
    return None
