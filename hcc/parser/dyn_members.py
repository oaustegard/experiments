"""Run-time offsets of members that follow a variably modified member (GNU).

    struct { int p[a], i; }   ->   offset(i) = align(a * 4, 4)

A struct is "dynamic" when some member is run-time sized (typesys.is_dyn_struct,
ty.dyn_tail). Members before the first run-time-sized one keep static offsets
(Member.offset); members from there on are in the dynamic region and their
offsets are built as unsigned long expressions at each use. The lengths are
the members' own VLA types, which vla_bind rebinds to hidden locals.

is_dyn_member_type(t)                 member type with a run-time size
first_dyn_index(ty) -> int | None     index of the first run-time-sized member
dyn_member_offset_expr(p, ty, i, loc) ulong expr: byte offset of members[i]
dyn_struct_size_expr(p, ty, loc)      ulong expr: sizeof a dynamic struct
dyn_member_access(p, node, ty, i, loc) lvalue of members[i] of struct value `node`
"""
from __future__ import annotations

import copy

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.arith import new_add
from hcc.sema.conversions import new_cast
from hcc.sema.dyn_struct_size import dyn_struct_bytes, dyn_walk, first_dyn_index


def in_dyn_region(ty, idx: int) -> bool:
    first = first_dyn_index(ty)
    return first is not None and idx >= first


def dyn_member_offset_expr(p, ty, idx: int, loc):
    """Byte offset of members[idx] (idx in the dynamic region), as a ulong expression."""
    return dyn_walk(ty, idx, loc, p.vla_size_expr)[0]


def dyn_struct_size_expr(p, ty, loc):
    """sizeof a struct with a run-time-sized member (see sema/dyn_struct_size.py)."""
    return dyn_struct_bytes(ty, loc, p.vla_size_expr)


def dyn_member_access(p, node, ty, idx: int, loc):
    """lvalue `node.members[idx]` for a member in the dynamic region: the object is
    addressed through its byte offset from the struct's address (struct values are
    addresses, see CONTRACT section 1). A bit-field becomes a member of offset 0 of the
    storage unit at that address, keeping its static bit_offset."""
    m = ty.members[idx]
    char_ptr = T.pointer_to(T.TY_CHAR)
    base = N.Cast(loc, node)
    base.ty = char_ptr
    addr = new_add(p.typed(base), dyn_member_offset_expr(p, ty, idx, loc), loc)
    p.typed(addr)
    obj_ptr = T.pointer_to(m.ty)
    cast = p.typed(new_cast(addr, obj_ptr))
    if not m.is_bitfield:
        return p.typed(N.Deref(loc, cast))
    unit = copy.copy(m)
    unit.offset = 0
    return N.Member_(loc, p.typed(N.Deref(loc, cast)), unit, ty=m.ty)
