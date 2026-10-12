"""Run-time layout of a struct with a run-time-sized member (GNU VLA tail).

Members from the first run-time-sized one on sit at run-time byte offsets. A bit-field
group in that region is anchored at a run-time byte offset and its bit-fields carry static
bit offsets (structs._dyn_bitfield). Used by parser/dyn_members.py (member offsets, sizeof)
and sema/arith.py (element strides); `size_of(ty, loc)` gives a member's run-time byte size.

dyn_walk(ty, upto, loc, size_of) -> (offset, end) ulong exprs of members[upto]
dyn_struct_bytes(ty, loc, size_of) ulong expr: sizeof a dynamic struct
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.typing import add_type


def _typed(node):
    add_type(node)
    return node


def _ulong(value: int, loc):
    return _typed(N.IntLit(loc, value, ty=T.TY_ULONG))


def _align_up(expr, align: int, loc):
    if align <= 1:
        return expr
    total = _typed(N.Binary(loc, "+", expr, _ulong(align - 1, loc)))
    return _typed(N.Binary(loc, "&", total, _ulong(-align & (2**64 - 1), loc)))


def _is_dyn_type(t) -> bool:
    while t is not None and t.kind == T.TypeKind.ARRAY:
        t = t.base
    if t is None:
        return False
    return (t.kind == T.TypeKind.VLA and t.vla_len is not None) or T.is_dyn_struct(t)


def first_dyn_index(ty):
    for i, m in enumerate(ty.members):
        if _is_dyn_type(m.ty):
            return i
    return None


def dyn_walk(ty, upto: int, loc, size_of):
    """(offset, end) ulong exprs of members[upto], which lies in the dynamic region."""
    first = first_dyn_index(ty)
    anchor = None
    end = None
    off = None
    for j in range(first, upto + 1):
        m = ty.members[j]
        if j == first:
            off = _ulong(m.offset, loc)
        elif m.is_bitfield and m.bit_offset:
            off = anchor
        else:
            off = _align_up(end, m.align, loc)
        if m.is_bitfield:
            anchor = off
            nbytes = (m.bit_offset + m.bit_width + 7) // 8
            end = _typed(N.Binary(loc, "+", off, _ulong(nbytes, loc)))
        else:
            end = _typed(N.Binary(loc, "+", off, size_of(m.ty, loc)))
    return off, end


def dyn_struct_bytes(ty, loc, size_of):
    """sizeof a struct with a run-time-sized member: end of the last member, rounded up."""
    last = len(ty.members) - 1
    first = first_dyn_index(ty)
    if last < first:
        return _ulong(ty.size, loc)
    end = dyn_walk(ty, last, loc, size_of)[1]
    return _align_up(end, ty.align, loc)
