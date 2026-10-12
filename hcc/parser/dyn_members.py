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

from hcc import nodes as N
from hcc import typesys as T
from hcc.parser.structs import is_dyn_member_type
from hcc.sema.arith import new_add
from hcc.sema.conversions import new_cast


def first_dyn_index(ty):
    for i, m in enumerate(ty.members):
        if is_dyn_member_type(m.ty):
            return i
    return None


def in_dyn_region(ty, idx: int) -> bool:
    first = first_dyn_index(ty)
    return first is not None and idx >= first


def _ulong(p, value: int, loc):
    return p.typed(N.IntLit(loc, value, ty=T.TY_ULONG))


def _align_up(p, expr, align: int, loc):
    if align <= 1:
        return expr
    total = p.typed(N.Binary(loc, "+", expr, _ulong(p, align - 1, loc)))
    return p.typed(N.Binary(loc, "&", total, _ulong(p, -align & (2**64 - 1), loc)))


def dyn_member_offset_expr(p, ty, idx: int, loc):
    """Byte offset of members[idx] (idx in the dynamic region), as a ulong expression."""
    first = first_dyn_index(ty)
    end = None
    for j in range(first, idx + 1):
        m = ty.members[j]
        if j == first:
            off = _ulong(p, m.offset, loc)
        else:
            off = _align_up(p, end, m.align, loc)
        if j == idx:
            return off
        size = p.vla_size_expr(m.ty, loc)
        end = p.typed(N.Binary(loc, "+", off, size))
    raise AssertionError("unreachable: idx is in the dynamic region")


def dyn_struct_size_expr(p, ty, loc):
    """sizeof a struct with a run-time-sized member: end of the last member, rounded up."""
    last = len(ty.members) - 1
    first = first_dyn_index(ty)
    if last < first:
        return _ulong(p, ty.size, loc)
    off = dyn_member_offset_expr(p, ty, last, loc)
    end = p.typed(N.Binary(loc, "+", off, p.vla_size_expr(ty.members[last].ty, loc)))
    return _align_up(p, end, ty.align, loc)


def dyn_member_access(p, node, ty, idx: int, loc):
    """lvalue `node.members[idx]` for a member in the dynamic region: the object is
    addressed through its byte offset from the struct's address (struct values are
    addresses, see CONTRACT section 1)."""
    m = ty.members[idx]
    char_ptr = T.pointer_to(T.TY_CHAR)
    base = N.Cast(loc, node)
    base.ty = char_ptr
    addr = new_add(p.typed(base), dyn_member_offset_expr(p, ty, idx, loc), loc)
    p.typed(addr)
    obj_ptr = T.pointer_to(m.ty)
    cast = p.typed(new_cast(addr, obj_ptr))
    return p.typed(N.Deref(loc, cast))
