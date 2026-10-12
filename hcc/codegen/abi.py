"""SysV x86-64 ABI classification helpers.

    has_flonum(ty, lo, hi, offset=0) -> bool   all scalars in [lo,hi) are float/double
    struct_in_regs(ty) -> (fp1, fp2) | None    how a <=16-byte struct is passed:
        None = in memory; fp1/fp2 = True if eightbyte 1/2 is SSE class
    has_unaligned(ty) -> bool                  packed struct with a misaligned member (MEMORY)
Long double and __int128 classification is still TODO (memory class for long double args).
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
    return offset < lo or hi <= offset or k in (TypeKind.FLOAT16, TypeKind.FLOAT, TypeKind.DOUBLE)


def has_ldouble(ty) -> bool:
    k = ty.kind
    if k in (TypeKind.STRUCT, TypeKind.UNION):
        return any(has_ldouble(m.ty) for m in ty.members)
    if k == TypeKind.ARRAY:
        return ty.array_len > 0 and has_ldouble(ty.base)
    return k == TypeKind.LDOUBLE


def has_unaligned(ty) -> bool:
    """True if a (packed) struct/union holds a non-bitfield member at an offset
    not aligned to its type: SysV classifies such aggregates as MEMORY."""
    if ty.kind in (TypeKind.STRUCT, TypeKind.UNION):
        for m in ty.members:
            if m.is_bitfield:
                continue
            if m.ty.align > 1 and m.offset % m.ty.align:
                return True
            if has_unaligned(m.ty):
                return True
        return False
    if ty.kind == TypeKind.ARRAY:
        return ty.array_len > 0 and has_unaligned(ty.base)
    return False


def is_x87_complex(ty) -> bool:
    """`_Complex long double` (SysV COMPLEX_X87): returned in st(0)=real, st(1)=imag;
    passed in memory like any other 32-byte aggregate."""
    return ty.is_complex and ty.base.kind == TypeKind.LDOUBLE


def is_memory_struct(ty) -> bool:
    """Struct/union returned via a hidden pointer (SysV MEMORY class)."""
    if is_x87_complex(ty):
        return False
    return ty.size > 16 or has_unaligned(ty)


def struct_arg_regs(ty):
    """Like struct_in_regs, but structs holding long double are memory class
    when passed as arguments (X87 eightbytes are never passed in registers)."""
    if has_ldouble(ty):
        return None
    return struct_in_regs(ty)


def struct_in_regs(ty):
    if ty.size > 16 or ty.size == 0 or has_unaligned(ty):
        return None
    fp1 = has_flonum(ty, 0, 8)
    fp2 = has_flonum(ty, 8, 16) if ty.size > 8 else False
    return fp1, fp2


def abi_kind(ty):
    """TypeKind used for call/return ABI decisions: GCC vector values are
    passed and returned like structs (their address is the value)."""
    return TypeKind.STRUCT if ty.is_vector else ty.kind


def eightbytes(ty) -> int:
    return 1 if ty.size <= 8 else 2
