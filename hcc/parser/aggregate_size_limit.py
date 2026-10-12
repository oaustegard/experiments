"""Maximum size of a struct/union type (GCC: "type 'struct S' is too large").

GCC rejects an aggregate whose size exceeds PTRDIFF_MAX (the largest object
size on LP64), e.g. `struct S { int a; char b[__LONG_MAX__]; };`.
"""
from __future__ import annotations

from hcc import typesys as T

# LP64 PTRDIFF_MAX: the largest permitted object / type size in bytes
OBJECT_SIZE_MAX = (1 << 63) - 1


def check_aggregate_size(parser, ty, where) -> None:
    """Diagnose a completed struct/union whose size exceeds OBJECT_SIZE_MAX."""
    if ty.size <= OBJECT_SIZE_MAX:
        return
    kw = "struct" if ty.kind == T.TypeKind.STRUCT else "union"
    name = ty.tag if ty.tag else "<anonymous>"
    parser.error("type '%s %s' is too large" % (kw, name), where)
