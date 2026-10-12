"""SysV x86-64 ABI classification helpers.

    has_flonum(ty, lo, hi, offset=0) -> bool   all scalars in [lo,hi) are float/double
    struct_in_regs(ty) -> (fp1, fp2) | None    how a <=16-byte struct is passed:
        None = in memory; fp1/fp2 = True if eightbyte 1/2 is SSE class
Long double, __int128 and unaligned/packed fields are TODO (memory class).
"""
from __future__ import annotations

from hcc.typesys import TypeKind


def has_flonum(ty, lo: int, hi: int, offset: int = 0) -> bool:
    k = ty.kind
    if k in (TypeKind.STRUCT, TypeKind.UNION):
        for m in ty.members:
            if not has_flonum(m.ty, lo, hi, offset + m.offset):
                return False
        return True
    if k == TypeKind.ARRAY:
        for i in range(max(ty.array_len, 0)):
            if not has_flonum(ty.base, lo, hi, offset + ty.base.size * i):
                return False
        return True
    return offset < lo or hi <= offset or k in (TypeKind.FLOAT, TypeKind.DOUBLE)


def struct_in_regs(ty):
    if ty.size > 16 or ty.size == 0:
        return None
    fp1 = has_flonum(ty, 0, 8)
    fp2 = has_flonum(ty, 8, 16) if ty.size > 8 else False
    return fp1, fp2


def eightbytes(ty) -> int:
    return 1 if ty.size <= 8 else 2
