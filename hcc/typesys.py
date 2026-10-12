"""C type representation for x86-64 LP64 (STABLE, CONTRACT section 4).

Rules:
  * Base singletons (TY_INT, ...) must never be mutated; use `t.copy()`.
  * Qualifiers (const/volatile/restrict/atomic) live on the Type object.
  * Struct/union types are mutable while being defined (forward tags).
"""
from __future__ import annotations

from enum import Enum


class TypeKind(Enum):
    VOID = "void"
    BOOL = "_Bool"
    CHAR = "char"
    SHORT = "short"
    INT = "int"
    LONG = "long"
    LLONG = "long long"
    INT128 = "__int128"
    FLOAT = "float"
    DOUBLE = "double"
    LDOUBLE = "long double"
    FLOAT128 = "_Float128"
    ENUM = "enum"
    PTR = "pointer"
    ARRAY = "array"
    VLA = "vla"
    FUNC = "function"
    STRUCT = "struct"
    UNION = "union"


INTEGER_KINDS = frozenset({TypeKind.BOOL, TypeKind.CHAR, TypeKind.SHORT, TypeKind.INT,
                           TypeKind.LONG, TypeKind.LLONG, TypeKind.INT128, TypeKind.ENUM})
FLOAT_KINDS = frozenset({TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LDOUBLE, TypeKind.FLOAT128})


class Member:
    """A struct/union member. Bitfields have is_bitfield=True."""

    __slots__ = ("name", "ty", "offset", "align", "idx", "is_bitfield",
                 "bit_offset", "bit_width", "tok")

    def __init__(self, name, ty, offset=0, align=1, idx=0):
        self.name: str | None = name
        self.ty: Type = ty
        self.offset = offset
        self.align = align
        self.idx = idx
        self.is_bitfield = False
        self.bit_offset = 0
        self.bit_width = 0
        self.tok = None

    def __repr__(self):
        return f"Member({self.name}, {self.ty}, off={self.offset})"


class Type:
    __slots__ = ("kind", "size", "align", "is_unsigned", "is_const", "is_volatile",
                 "is_restrict", "is_atomic", "base", "array_len", "vla_len", "vla_size",
                 "members", "tag", "is_flexible", "is_packed", "is_complete",
                 "return_ty", "params", "param_names", "is_variadic", "has_proto",
                 "origin", "name", "enum_consts")

    def __init__(self, kind: TypeKind, size: int = 0, align: int = 1, is_unsigned: bool = False):
        self.kind = kind
        self.size = size
        self.align = align
        self.is_unsigned = is_unsigned
        self.is_const = False
        self.is_volatile = False
        self.is_restrict = False
        self.is_atomic = False
        self.base: Type | None = None
        self.array_len = 0
        self.vla_len = None
        self.vla_size = None
        self.members: list[Member] = []
        self.tag: str | None = None
        self.is_flexible = False
        self.is_packed = False
        self.is_complete = True
        self.return_ty: Type | None = None
        self.params: list[Type] = []
        self.param_names: list = []
        self.is_variadic = False
        self.has_proto = True
        self.origin: Type | None = None
        self.name = None          # declarator name Token while parsing
        self.enum_consts = None

    def copy(self) -> "Type":
        t = Type.__new__(Type)
        for s in Type.__slots__:
            setattr(t, s, getattr(self, s))
        t.origin = self
        return t

    def unqual(self) -> "Type":
        if not (self.is_const or self.is_volatile or self.is_restrict or self.is_atomic):
            return self
        t = self.copy()
        t.is_const = t.is_volatile = t.is_restrict = t.is_atomic = False
        return t

    def __repr__(self) -> str:
        return type_str(self)


def type_str(t: Type) -> str:
    q = "const " if t.is_const else ""
    k = t.kind
    if k == TypeKind.PTR:
        return f"{type_str(t.base)} *{'const' if t.is_const else ''}".rstrip()
    if k == TypeKind.ARRAY:
        return f"{type_str(t.base)}[{t.array_len if t.array_len >= 0 else ''}]"
    if k == TypeKind.VLA:
        return f"{type_str(t.base)}[*]"
    if k == TypeKind.FUNC:
        ps = ", ".join(type_str(p) for p in t.params) + (", ..." if t.is_variadic else "")
        return f"{type_str(t.return_ty)}({ps})"
    if k in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ENUM):
        return f"{q}{k.value} {t.tag or '<anon>'}"
    u = "unsigned " if t.is_unsigned and k not in (TypeKind.BOOL,) else ""
    return f"{q}{u}{k.value}"


def _base(kind, size, align, unsigned=False) -> Type:
    return Type(kind, size, align, unsigned)


TY_VOID = _base(TypeKind.VOID, 1, 1)
TY_BOOL = _base(TypeKind.BOOL, 1, 1, True)
TY_CHAR = _base(TypeKind.CHAR, 1, 1)
TY_SHORT = _base(TypeKind.SHORT, 2, 2)
TY_INT = _base(TypeKind.INT, 4, 4)
TY_LONG = _base(TypeKind.LONG, 8, 8)
TY_LLONG = _base(TypeKind.LLONG, 8, 8)
TY_INT128 = _base(TypeKind.INT128, 16, 16)
TY_UCHAR = _base(TypeKind.CHAR, 1, 1, True)
TY_USHORT = _base(TypeKind.SHORT, 2, 2, True)
TY_UINT = _base(TypeKind.INT, 4, 4, True)
TY_ULONG = _base(TypeKind.LONG, 8, 8, True)
TY_ULLONG = _base(TypeKind.LLONG, 8, 8, True)
TY_UINT128 = _base(TypeKind.INT128, 16, 16, True)
TY_FLOAT = _base(TypeKind.FLOAT, 4, 4)
TY_DOUBLE = _base(TypeKind.DOUBLE, 8, 8)
TY_LDOUBLE = _base(TypeKind.LDOUBLE, 16, 16)
TY_FLOAT128 = _base(TypeKind.FLOAT128, 16, 16)


def is_integer(t: Type) -> bool:
    return t.kind in INTEGER_KINDS


def is_flonum(t: Type) -> bool:
    return t.kind in FLOAT_KINDS


def is_numeric(t: Type) -> bool:
    return is_integer(t) or is_flonum(t)


def is_pointer_like(t: Type) -> bool:
    """Pointer, array or VLA (anything with a `base` you can index)."""
    return t.kind in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.VLA)


def is_scalar(t: Type) -> bool:
    return is_numeric(t) or t.kind == TypeKind.PTR


def is_aggregate(t: Type) -> bool:
    return t.kind in (TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY)


def is_struct_like(t: Type) -> bool:
    return t.kind in (TypeKind.STRUCT, TypeKind.UNION)


def pointer_to(base: Type) -> Type:
    t = Type(TypeKind.PTR, 8, 8, True)
    t.base = base
    return t


def array_of(base: Type, n: int) -> Type:
    t = Type(TypeKind.ARRAY, base.size * n if n >= 0 else 0, base.align)
    t.base = base
    t.array_len = n
    t.is_complete = n >= 0
    return t


def vla_of(base: Type, length_expr) -> Type:
    t = Type(TypeKind.VLA, 8, 8)
    t.base = base
    t.vla_len = length_expr
    return t


def func_type(return_ty: Type) -> Type:
    t = Type(TypeKind.FUNC, 1, 1)
    t.return_ty = return_ty
    return t


def struct_type(kind: TypeKind = TypeKind.STRUCT) -> Type:
    t = Type(kind, 0, 1)
    t.is_complete = False
    return t


def enum_type() -> Type:
    t = Type(TypeKind.ENUM, 4, 4, True)
    return t


def va_list_type() -> Type:
    """__builtin_va_list: array[1] of the 24-byte SysV va_list struct."""
    s = struct_type()
    s.tag = "__va_list_tag"
    s.members = [Member("gp_offset", TY_UINT, 0, 4, 0), Member("fp_offset", TY_UINT, 4, 4, 1),
                 Member("overflow_arg_area", pointer_to(TY_VOID), 8, 8, 2),
                 Member("reg_save_area", pointer_to(TY_VOID), 16, 8, 3)]
    s.size, s.align, s.is_complete = 24, 8, True
    return array_of(s, 1)


def int_type_for(size: int, unsigned: bool) -> Type:
    table = {1: (TY_CHAR, TY_UCHAR), 2: (TY_SHORT, TY_USHORT), 4: (TY_INT, TY_UINT),
             8: (TY_LONG, TY_ULONG), 16: (TY_INT128, TY_UINT128)}
    return table[size][1 if unsigned else 0]


def align_to(n: int, align: int) -> int:
    return (n + align - 1) // align * align if align > 1 else n
