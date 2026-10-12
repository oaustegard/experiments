"""Read-modify-write lowering of `a op= b`, `++a`, `a++` on an _Atomic integer
object (C11 7.17.7.5: the update is one atomic operation).

The expression becomes the gcc __atomic_{add,sub,and,or,xor}_fetch builtin
(new value) or __atomic_fetch_add (old value, for the post forms), which
codegen/atomics.py lowers to a lock cmpxchg loop.  Operators without a single
x86 RMW form (*, /, %, <<, >>) and non-integer operands return None and keep
the plain lowering in sema/arith.py.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.conversions import new_cast
from hcc.sema.typing import add_type
from hcc.typesys import TypeKind, is_integer

_RMW_OPS = {"+": "add", "-": "sub", "&": "and", "|": "or", "^": "xor"}


def _is_atomic_int_object(lhs) -> bool:
    ty = lhs.ty
    if ty is None or not ty.is_atomic or not is_integer(ty) or ty.kind is TypeKind.BOOL:
        return False
    return not (isinstance(lhs, N.Member_) and lhs.member.is_bitfield)


def atomic_rmw(op: str, lhs, rhs, loc, want_old: bool):
    """Lowered expression for `lhs op rhs` on an _Atomic integer lhs, or None.

    want_old: the value of the expression is the object's value before the update
    (post-increment); otherwise it is the value after the update."""
    name = _RMW_OPS.get(op)
    if name is None or not _is_atomic_int_object(lhs):
        return None
    add_type(rhs)
    if rhs.ty is None or not is_integer(rhs.ty):
        return None
    ty = lhs.ty.unqual()
    ptr = N.AddrOf(loc, lhs)
    add_type(ptr)
    val = new_cast(rhs, ty)
    builtin = f"atomic_fetch_{name}" if want_old else f"atomic_{name}_fetch"
    return N.BuiltinCall(loc, builtin, [ptr, val], None, ty=ty)
