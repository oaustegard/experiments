"""Scalar conversions (chibicc cast table, with fixes).

CastMixin.cast(from_ty, to_ty) converts the accumulator in place.
Type ids: i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 f80. Pointers are u64,
_Bool target is handled by cmp_zero. Add new conversions to TABLE only.
"""
from __future__ import annotations

from hcc.typesys import TypeKind

I8, I16, I32, I64, U8, U16, U32, U64, F32, F64, F80 = range(11)

i32i8 = "movsbl %al, %eax"
i32u8 = "movzbl %al, %eax"
i32i16 = "movswl %ax, %eax"
i32u16 = "movzwl %ax, %eax"
i32f32 = "cvtsi2ssl %eax, %xmm0"
i32i64 = "movslq %eax, %rax"
i32f64 = "cvtsi2sdl %eax, %xmm0"
i32f80 = "mov %eax, -4(%rsp); fildl -4(%rsp)"
u32f32 = "mov %eax, %eax; cvtsi2ssq %rax, %xmm0"
u32i64 = "mov %eax, %eax"
u32f64 = "mov %eax, %eax; cvtsi2sdq %rax, %xmm0"
u32f80 = "mov %eax, %eax; mov %rax, -8(%rsp); fildll -8(%rsp)"
i64f32 = "cvtsi2ssq %rax, %xmm0"
i64f64 = "cvtsi2sdq %rax, %xmm0"
i64f80 = "movq %rax, -8(%rsp); fildll -8(%rsp)"
u64f32 = ("test %rax,%rax; js 1f; pxor %xmm0,%xmm0; cvtsi2ssq %rax,%xmm0; jmp 2f; "
          "1: mov %rax,%rdi; and $1,%eax; pxor %xmm0,%xmm0; shr %rdi; "
          "or %rax,%rdi; cvtsi2ssq %rdi,%xmm0; addss %xmm0,%xmm0; 2:")
u64f64 = ("test %rax,%rax; js 1f; pxor %xmm0,%xmm0; cvtsi2sdq %rax,%xmm0; jmp 2f; "
          "1: mov %rax,%rdi; and $1,%eax; pxor %xmm0,%xmm0; shr %rdi; "
          "or %rax,%rdi; cvtsi2sdq %rdi,%xmm0; addsd %xmm0,%xmm0; 2:")
u64f80 = ("mov %rax, -8(%rsp); fildq -8(%rsp); test %rax, %rax; jns 1f;"
          "mov $1602224128, %eax; mov %eax, -4(%rsp); fadds -4(%rsp); 1:")


def _from_flt(sfx: str):
    cv = f"cvtt{sfx}2si"
    # float/double -> uint64: values >= 2^63 do not fit cvtt...2siq, so
    # subtract 2^63, convert, and flip bit 63 back (as GCC does).
    if sfx == "ss":
        load, ucomi, sub = "mov $0x5f000000, %ecx; movd %ecx, %xmm1", "ucomiss", "subss"
    else:
        load, ucomi, sub = "movabs $0x43e0000000000000, %rcx; movq %rcx, %xmm1", "ucomisd", "subsd"
    u64 = (f"{load}; {ucomi} %xmm1, %xmm0; jae 1f; {cv}q %xmm0, %rax; jmp 2f; "
           f"1: {sub} %xmm1, %xmm0; {cv}q %xmm0, %rax; btc $63, %rax; 2:")
    return {
        I8: f"{cv}l %xmm0, %eax; movsbl %al, %eax", U8: f"{cv}l %xmm0, %eax; movzbl %al, %eax",
        I16: f"{cv}l %xmm0, %eax; movswl %ax, %eax", U16: f"{cv}l %xmm0, %eax; movzwl %ax, %eax",
        I32: f"{cv}l %xmm0, %eax", U32: f"{cv}q %xmm0, %rax", I64: f"{cv}q %xmm0, %rax",
        U64: u64,
    }


_F32 = _from_flt("ss")
_F32.update({F64: "cvtss2sd %xmm0, %xmm0", F80: "movss %xmm0, -4(%rsp); flds -4(%rsp)"})
_F64 = _from_flt("sd")
_F64.update({F32: "cvtsd2ss %xmm0, %xmm0", F80: "movsd %xmm0, -8(%rsp); fldl -8(%rsp)"})

_F80_1 = "fnstcw -10(%rsp); movzwl -10(%rsp), %eax; or $12, %ah; mov %ax, -12(%rsp); fldcw -12(%rsp); "
_F80_2 = " -24(%rsp); fldcw -10(%rsp); "
_F80 = {
    I8: _F80_1 + "fistps" + _F80_2 + "movsbl -24(%rsp), %eax",
    U8: _F80_1 + "fistps" + _F80_2 + "movzbl -24(%rsp), %eax",
    I16: _F80_1 + "fistps" + _F80_2 + "movswl -24(%rsp), %eax",
    U16: _F80_1 + "fistpl" + _F80_2 + "movzwl -24(%rsp), %eax",
    I32: _F80_1 + "fistpl" + _F80_2 + "mov -24(%rsp), %eax",
    U32: _F80_1 + "fistpq" + _F80_2 + "mov -24(%rsp), %eax",
    I64: _F80_1 + "fistpq" + _F80_2 + "mov -24(%rsp), %rax",
    U64: _F80_1 + "fistpq" + _F80_2 + "mov -24(%rsp), %rax",
    F32: "fstps -8(%rsp); movss -8(%rsp), %xmm0",
    F64: "fstpl -8(%rsp); movsd -8(%rsp), %xmm0",
}

N_ = None
TABLE = [
    # i8     i16     i32  i64     u8     u16     u32  u64     f32     f64     f80
    [N_,     N_,     N_,  i32i64, i32u8, i32u16, N_,  i32i64, i32f32, i32f64, i32f80],  # i8
    [i32i8,  N_,     N_,  i32i64, i32u8, i32u16, N_,  i32i64, i32f32, i32f64, i32f80],  # i16
    [i32i8,  i32i16, N_,  i32i64, i32u8, i32u16, N_,  i32i64, i32f32, i32f64, i32f80],  # i32
    [i32i8,  i32i16, N_,  N_,     i32u8, i32u16, N_,  N_,     i64f32, i64f64, i64f80],  # i64
    [i32i8,  N_,     N_,  i32i64, N_,    N_,     N_,  i32i64, i32f32, i32f64, i32f80],  # u8
    [i32i8,  i32i16, N_,  i32i64, i32u8, N_,     N_,  i32i64, i32f32, i32f64, i32f80],  # u16
    [i32i8,  i32i16, N_,  u32i64, i32u8, i32u16, N_,  u32i64, u32f32, u32f64, u32f80],  # u32
    [i32i8,  i32i16, N_,  N_,     i32u8, i32u16, N_,  N_,     u64f32, u64f64, u64f80],  # u64
    [_F32[I8], _F32[I16], _F32[I32], _F32[I64], _F32[U8], _F32[U16], _F32[U32], _F32[U64], N_, _F32[F64], _F32[F80]],
    [_F64[I8], _F64[I16], _F64[I32], _F64[I64], _F64[U8], _F64[U16], _F64[U32], _F64[U64], _F64[F32], N_, _F64[F80]],
    [_F80[I8], _F80[I16], _F80[I32], _F80[I64], _F80[U8], _F80[U16], _F80[U32], _F80[U64], _F80[F32], _F80[F64], N_],
]


def type_id(ty) -> int:
    k = ty.kind
    if k == TypeKind.FLOAT:
        return F32
    if k == TypeKind.DOUBLE:
        return F64
    if k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
        return F80
    u = ty.is_unsigned
    sz = ty.size
    if k in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.FUNC, TypeKind.STRUCT, TypeKind.UNION):
        return U64
    if sz == 1:
        return U8 if u else I8
    if sz == 2:
        return U16 if u else I16
    if sz == 4:
        return U32 if u else I32
    return U64 if u else I64


class CastMixin:

    def cmp_zero(self, ty) -> None:
        k = ty.kind
        if k == TypeKind.FLOAT:
            self.emit("xorps %xmm1, %xmm1")
            self.emit("ucomiss %xmm1, %xmm0")
        elif k == TypeKind.DOUBLE:
            self.emit("xorpd %xmm1, %xmm1")
            self.emit("ucomisd %xmm1, %xmm0")
        elif k in (TypeKind.LDOUBLE, TypeKind.FLOAT128):
            self.emit("fldz")
            self.emit("fucomip")
            self.emit("fstp %st(0)")
        elif k == TypeKind.INT128:
            self.int128_cmp_zero()
        elif ty.size <= 4 and k not in (TypeKind.PTR, TypeKind.ARRAY, TypeKind.FUNC):
            self.emit("cmp $0, %eax")
        else:
            self.emit("cmp $0, %rax")

    def cast(self, frm, to) -> None:
        if to.kind == TypeKind.VOID:
            return
        if to.kind == TypeKind.BOOL:
            self.cmp_zero(frm)
            self.emit("setne %al")
            self.emit("movzbl %al, %eax")
            return
        if frm.kind in (TypeKind.STRUCT, TypeKind.UNION) or to.kind in (TypeKind.STRUCT, TypeKind.UNION):
            return
        if frm.kind == TypeKind.VOID:
            return
        if frm.kind == TypeKind.INT128 or to.kind == TypeKind.INT128:
            if self.int128_cast(frm, to):
                return
        code = TABLE[type_id(frm)][type_id(to)]
        if code:
            for part in code.split("; "):
                self.emit(part)
