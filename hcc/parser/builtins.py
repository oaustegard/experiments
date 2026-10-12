"""Parsing of special __builtin_* forms (those taking types, or needing
compile-time evaluation). Plain library builtins such as __builtin_memcpy
are handled in primary.py by calling the libc function of the same name.

BuiltinParseMixin.parse_builtin(name_tok) -> Expr
To add one: put the name in SPECIAL_BUILTINS and add a `_b_<name>` method
(name without the `__builtin_` prefix); lower it in codegen/builtins.py if it
produces a BuiltinCall.
"""
from __future__ import annotations

import math
import struct

from hcc import nodes as N
from hcc.halfbits import half_to_double
from hcc.parser.fp_classify import FP_CLASS_STEMS
from hcc.parser.float16_math import F16_MATH_NAMES, f16_math_call
from hcc.parser.atomic_builtins import ATOMIC_BUILTINS, atomic_call
from hcc.parser.crc_builtins import CRC_BUILTINS, crc_call
from hcc.parser.object_size import lower_object_size
from hcc.parser.structs import find_member
from hcc.parser.overflow_wide import wide_overflow_operands
from hcc.parser.vector_cast import lower_convertvector
from hcc.parser.vector_shuffle import lower_shuffle
from hcc.parser.vector_shufflevector import lower_shufflevector
from hcc.parser.tgmath import tgmath_call
from hcc.sema.compat import is_compatible
from hcc.sema.constexpr import builtin_constant_p, try_eval_const
from hcc.sema.conversions import new_cast
from hcc import typesys as T

SPECIAL_BUILTINS = frozenset("""
__builtin_va_start __builtin_va_end __builtin_va_arg __builtin_va_copy __builtin_offsetof
__builtin_va_arg_pack __builtin_va_arg_pack_len
__builtin_tgmath
__builtin_types_compatible_p __builtin_expect __builtin_constant_p __builtin_unreachable
__builtin_trap __builtin_alloca __builtin_alloca_with_align __builtin_choose_expr __builtin_classify_type
__builtin_call_with_static_chain
__builtin_add_overflow __builtin_sub_overflow __builtin_mul_overflow __builtin_mul_overflow_p
__builtin_add_overflow_p __builtin_sub_overflow_p
__builtin_prefetch __builtin_return_address __builtin_frame_address __builtin_assoc_barrier
__builtin_assume_aligned
__builtin_bswap16 __builtin_bswap32 __builtin_bswap64
__builtin_huge_val __builtin_huge_valf __builtin_huge_vall
__builtin_inf __builtin_inff __builtin_infl __builtin_signbit __builtin_signbitf __builtin_signbitl
__builtin_nan __builtin_nanf __builtin_nanl __builtin_nans __builtin_nansf __builtin_nansl
__builtin_setjmp __builtin_longjmp __builtin_shuffle __builtin_shufflevector __builtin_convertvector
__builtin_isinf __builtin_isnan __builtin_isfinite __builtin_isnormal
__builtin_isinff __builtin_isinfl __builtin_isnanf __builtin_isnanl __builtin_isfinitef __builtin_isfinitel
__builtin_finite __builtin_finitef __builtin_finitel
__builtin_isunordered __builtin_isgreater __builtin_isgreaterequal
__builtin_fpclassify __builtin_isnormal __builtin_isinf_sign
__builtin_isless __builtin_islessequal __builtin_islessgreater
__builtin_ia32_crc32qi __builtin_ia32_crc32hi __builtin_ia32_crc32si __builtin_ia32_crc32di
__builtin_object_size __builtin_dynamic_object_size
""".split())

# SSE4.2 CRC32 builtins: (crc, value) -> CRC-32C accumulated over the value's width.
# The value operand width gives the codegen suffix; `di` works on 64-bit operands.
CRC32_BUILTINS = {"qi": T.TY_UCHAR, "hi": T.TY_USHORT, "si": T.TY_UINT, "di": T.TY_ULLONG}

# Bit-counting builtins: {clz,ctz,popcount,parity,ffs}{,l,ll}. The suffix picks
# the operand width (int = 32 bit, long/long long = 64 bit).
BIT_BUILTINS = frozenset(
    f"__builtin_{base}{suffix}"
    for base in ("clz", "ctz", "popcount", "parity", "ffs", "clrsb")
    for suffix in ("", "l", "ll"))
# GCC's _FloatN spellings of the inf/nan builtins, e.g. __builtin_inff64x (the
# _Float64x variant). Mapped onto the base builtin of the matching C type width:
# _Float32 -> float, _Float32x/_Float64 -> double, _Float64x/_Float128 -> long double.
_FLOATN_SUFFIX = {"32": "f", "32x": "", "64": "", "64x": "l", "128": "l", "128x": "l"}
FLOATN_BUILTIN_MAP = {
    f"__builtin_{base}f{width}": f"__builtin_{base}{sfx}"
    for base in ("inf", "huge_val", "nan", "nans")
    for width, sfx in _FLOATN_SUFFIX.items()}

# _Float16 forms (__builtin_inff16, __builtin_sinf16, ...): see _float16_builtin.
FLOAT16_BUILTINS = frozenset(f"__builtin_{base}f16" for base in
                             ("inf", "huge_val", "nan", "nans") + F16_MATH_NAMES)
SPECIAL_BUILTINS = (SPECIAL_BUILTINS | BIT_BUILTINS | ATOMIC_BUILTINS | frozenset(FLOATN_BUILTIN_MAP)
                    | FLOAT16_BUILTINS | CRC_BUILTINS)


def floatn_libc_name(name: str) -> str:
    """`fabsf64x` -> `fabsl`, `acoshf32` -> `acoshf`: the libm function behind a GCC
    __builtin_<fn>f<N>[x] spelling (the libm suffix of the same C type width).
    Other names are returned unchanged."""
    for width, sfx in _FLOATN_SUFFIX.items():
        tail = "f" + width
        if name.endswith(tail) and len(name) > len(tail):
            return name[:-len(tail)] + sfx
    return name


class BuiltinParseMixin:

    def parse_builtin(self, tok):
        if tok.text in ATOMIC_BUILTINS:
            self.expect("(")
            node = atomic_call(self, tok)
            self.expect(")")
            return node
        if tok.text in FLOAT16_BUILTINS:
            self.expect("(")
            node = self._float16_builtin(tok)
            self.expect(")")
            return node
        name = FLOATN_BUILTIN_MAP.get(tok.text, tok.text)[len("__builtin_"):]
        self.expect("(")
        if tok.text in BIT_BUILTINS:
            node = self._bitop(tok, name)
        elif tok.text in CRC_BUILTINS:
            node = crc_call(self, tok)
        else:
            node = getattr(self, "_b_" + name)(tok)
        self.expect(")")
        return node

    def _b_tgmath(self, tok):
        return tgmath_call(self, tok)

    def _float16_builtin(self, tok):
        """__builtin_{inf,huge_val,nan,nans}f16(...): a _Float16 constant. The NaN bit
        patterns are built directly in binary16 (a float or double round trip would
        quiet a signaling NaN). Other math functions go through parser/float16_math.py."""
        base = tok.text[len("__builtin_"):-len("f16")]
        if base in F16_MATH_NAMES:
            return f16_math_call(self, tok, base)
        if base.startswith("nan"):
            self._args(1)
            val = half_to_double(0x7D01 if base == "nans" else 0x7E00)
        else:
            self._args(0)
            val = math.inf
        return N.FloatLit(tok.loc, val, ty=T.TY_FLOAT16)

    def _bitop(self, tok, name):
        """clz/ctz/popcount/parity/ffs: argument converted to unsigned int
        (or unsigned long for the l/ll forms); result is int."""
        base = name.rstrip("l")
        bits = 64 if name[len(base):] else 32
        ty = T.TY_ULONG if bits == 64 else T.TY_UINT
        if base in ("ffs", "clrsb"):  # signed operands
            ty = T.TY_LONG if bits == 64 else T.TY_INT
        args = [new_cast(a, ty) for a in self._args(1)]
        return self._call(tok, f"{base}{bits}", args, T.TY_INT)

    def _args(self, n=None):
        args = []
        while not self.at(")"):
            if args:
                self.expect(",")
            args.append(self.typed(self.assign()))
        if n is not None and len(args) != n:
            self.error(f"builtin expects {n} arguments")
        return args

    def _call(self, tok, name, args, ty, type_arg=None):
        return N.BuiltinCall(tok.loc, name, args, type_arg, ty=ty)

    def _b_shuffle(self, tok):
        return lower_shuffle(self, tok.loc, self._args())

    def _b_shufflevector(self, tok):
        return lower_shufflevector(self, tok.loc, self._args())

    def _b_convertvector(self, tok):
        """__builtin_convertvector(vec, vector_type): element-wise conversion."""
        src = self.typed(self.assign())
        self.expect(",")
        dst = self.typename()
        return lower_convertvector(self, tok.loc, src, dst)

    def _b_va_start(self, tok):
        args = self._args()
        return self._call(tok, "va_start", args[:1], T.TY_VOID)

    def _b_va_arg_pack(self, tok):
        """Placeholder resolved by parser/va_pack.py (extra args of the inlined call)."""
        return self._call(tok, "va_arg_pack", [], T.TY_INT)

    def _b_va_arg_pack_len(self, tok):
        return self._call(tok, "va_arg_pack_len", [], T.TY_INT)

    def _b_va_end(self, tok):
        self._args()
        return N.Cast(tok.loc, N.IntLit(tok.loc, 0, ty=T.TY_INT), ty=T.TY_VOID)

    def _b_va_copy(self, tok):
        return self._call(tok, "va_copy", self._args(2), T.TY_VOID)

    def _b_va_arg(self, tok):
        ap = self.typed(self.assign())
        self.expect(",")
        ty = self.typename()
        # a va_list may be consumed by any function (not only variadic ones,
        # e.g. `void f(va_list ap)`), so make sure the split-struct scratch exists
        fn = getattr(self, "fn", None)
        if fn is not None and fn.va_tmp is None:
            fn.va_tmp = self.new_temp(T.array_of(T.TY_CHAR, 16))
        return self._call(tok, "va_arg", [ap], ty, ty)

    def _b_shuffle(self, tok):
        """__builtin_shuffle(v, mask) / __builtin_shuffle(a, b, mask): element
        i of the result is v[mask[i] % N] (or the concatenation a:b for the
        3-argument form, indices taken modulo 2N). Result type is that of the
        first vector."""
        args = self._args()
        if len(args) not in (2, 3):
            self.error("builtin expects 2 or 3 arguments")
        a, mask = args[0], args[-1]
        if not a.ty.is_vector or not mask.ty.is_vector or not T.is_integer(mask.ty.base):
            self.error("'__builtin_shuffle' expects integer vector operands")
        n = a.ty.size // a.ty.base.size
        if mask.ty.size // mask.ty.base.size != n:
            self.error("'__builtin_shuffle' mask must have as many elements as the vector")
        if len(args) == 3 and not is_compatible(args[1].ty, a.ty):
            self.error("'__builtin_shuffle' vector operands must have the same type")
        node = self._call(tok, "shuffle", args, a.ty)
        node.result_buf = self.new_temp(a.ty, tok.loc)
        return node

    def _b_prefetch(self, tok):
        """__builtin_prefetch(addr[, rw[, locality]]) is only a cache hint: it
        evaluates its arguments (for side effects) and yields void."""
        args = self._args()
        if not 1 <= len(args) <= 3:
            self.error("builtin expects 1 to 3 arguments")
        node = N.Cast(tok.loc, N.IntLit(tok.loc, 0, ty=T.TY_INT), ty=T.TY_VOID)
        for a in reversed(args):
            node = N.Comma(tok.loc, N.Cast(tok.loc, a, ty=T.TY_VOID), node, ty=T.TY_VOID)
        return node

    def _frame_level(self, tok, name):
        """__builtin_{return,frame}_address(n): n must be a non-negative integer
        constant. The level is passed as an IntLit; codegen walks the %rbp chain."""
        args = self._args(1)
        level = try_eval_const(args[0])
        if level is None or level < 0:
            self.error(f"'__builtin_{name}' needs a non-negative integer constant")
        return self._call(tok, name, [N.IntLit(tok.loc, level, ty=T.TY_INT)],
                          T.pointer_to(T.TY_VOID))

    def _b_return_address(self, tok):
        return self._frame_level(tok, "return_address")

    def _b_frame_address(self, tok):
        return self._frame_level(tok, "frame_address")

    def _bswap(self, tok, bits, ty):
        args = [new_cast(a, ty) for a in self._args(1)]
        return self._call(tok, f"bswap{bits}", args, ty)

    def _b_bswap16(self, tok):
        return self._bswap(tok, 16, T.TY_USHORT)

    def _b_bswap32(self, tok):
        return self._bswap(tok, 32, T.TY_UINT)

    def _b_bswap64(self, tok):
        return self._bswap(tok, 64, T.TY_ULONG)

    def _crc32(self, tok, width):
        """__builtin_ia32_crc32{qi,hi,si,di}(crc, v): crc is unsigned int (unsigned
        long long for di); the result has the same type as crc."""
        crc_ty = T.TY_ULLONG if width == "di" else T.TY_UINT
        crc, val = self._args(2)
        args = [new_cast(crc, crc_ty), new_cast(val, CRC32_BUILTINS[width])]
        return self._call(tok, f"ia32_crc32{width}", args, crc_ty)

    def _b_ia32_crc32qi(self, tok):
        return self._crc32(tok, "qi")

    def _b_ia32_crc32hi(self, tok):
        return self._crc32(tok, "hi")

    def _b_ia32_crc32si(self, tok):
        return self._crc32(tok, "si")

    def _b_ia32_crc32di(self, tok):
        return self._crc32(tok, "di")

    def _b_alloca(self, tok):
        args = [new_cast(a, T.TY_ULONG) for a in self._args(1)]
        return self._call(tok, "alloca", args, T.pointer_to(T.TY_VOID))

    def _b_alloca_with_align(self, tok):
        """__builtin_alloca_with_align(size, align_in_bits): align must be a constant.
        The second argument of the lowered call is the alignment in bytes."""
        size, align = self._args(2)
        align_val = try_eval_const(align)
        byte_align = (align_val or 0) // 8
        if byte_align <= 0 or byte_align & (byte_align - 1):
            self.error("__builtin_alloca_with_align: alignment must be a power of two (in bits)")
        args = [new_cast(size, T.TY_ULONG), N.IntLit(tok.loc, byte_align, ty=T.TY_ULONG)]
        return self._call(tok, "alloca_align", args, T.pointer_to(T.TY_VOID))

    def _b_setjmp(self, tok):
        """__builtin_setjmp(buf): buf is a void *[5]-style record; returns 0."""
        args = [new_cast(a, T.pointer_to(T.TY_VOID)) for a in self._args(1)]
        return self._call(tok, "setjmp", args, T.TY_INT)

    def _b_longjmp(self, tok):
        """__builtin_longjmp(buf, 1): resumes the matching __builtin_setjmp."""
        args = self._args(2)
        args = [new_cast(args[0], T.pointer_to(T.TY_VOID)), new_cast(args[1], T.TY_INT)]
        return self._call(tok, "longjmp", args, T.TY_VOID)

    def _overflow(self, tok, name):
        """__builtin_{add,sub,mul}_overflow(a, b, &res): operands are widened to
        exact 64-bit LONG/ULONG (by their own signedness); codegen does the check."""
        args = self._args(3)
        res = args[2]
        if res.ty.kind != T.TypeKind.PTR or res.ty.base is None:
            self.error(f"'__builtin_{name}_overflow' expects a pointer as third argument")
        ops = wide_overflow_operands(self, name, args[:2], res.ty.base)
        if ops is None:
            ops = [new_cast(a, T.TY_ULONG if a.ty.is_unsigned else T.TY_LONG) for a in args[:2]]
        return self._call(tok, name + "_overflow", ops + [res], T.TY_INT, res.ty.base)

    def _b_add_overflow(self, tok):
        return self._overflow(tok, "add")

    def _b_sub_overflow(self, tok):
        return self._overflow(tok, "sub")

    def _b_mul_overflow(self, tok):
        return self._overflow(tok, "mul")

    def _overflow_p(self, tok, name):
        """__builtin_{add,sub,mul}_overflow_p(a, b, c): 1 iff a OP b does not fit
        the type of c. Only the type of c is used (its value is never read or stored)."""
        args = self._args(3)
        ops = wide_overflow_operands(self, name, args[:2], args[2].ty)
        if ops is None:
            ops = [new_cast(a, T.TY_ULONG if a.ty.is_unsigned else T.TY_LONG) for a in args[:2]]
        return self._call(tok, name + "_overflow_p", ops, T.TY_INT, args[2].ty)

    def _b_mul_overflow_p(self, tok):
        return self._overflow_p(tok, "mul")

    def _b_add_overflow_p(self, tok):
        return self._overflow_p(tok, "add")

    def _b_sub_overflow_p(self, tok):
        return self._overflow_p(tok, "sub")

    def _b_object_size(self, tok):
        return lower_object_size(self, tok)

    def _b_dynamic_object_size(self, tok):
        return lower_object_size(self, tok)

    def _b_assoc_barrier(self, tok):
        # Identity on the value: it only stops reassociation of floating expressions.
        return self._args(1)[0]

    def _b_assume_aligned(self, tok):
        """__builtin_assume_aligned(p, align[, offset]) yields p unchanged (the
        alignment is only an optimisation hint); the other arguments are evaluated
        for their side effects, as in __builtin_prefetch."""
        args = self._args()
        if not 2 <= len(args) <= 3:
            self.error("builtin expects 2 or 3 arguments")
        node = args[0]
        for a in reversed(args[1:]):
            node = N.Comma(tok.loc, N.Cast(tok.loc, a, ty=T.TY_VOID), node, ty=node.ty)
        return node

    def _b_call_with_static_chain(self, tok):
        # (call, chain): perform `call` with the static chain register (%r10) set to `chain`.
        args = self._args(2)
        call = args[0]
        if type(call) is not N.Call:
            self.error("first argument of __builtin_call_with_static_chain must be a call")
        slot = self.new_temp(args[1].ty, tok.loc)
        call.chain_slot = slot
        call.static_chain = self.typed(N.Assign(tok.loc, self.typed(N.VarRef(tok.loc, slot)), args[1]))
        return call

    def _b_expect(self, tok):
        args = self._args(2)
        # The second argument (the expected value) is still evaluated for its
        # side effects; the result is the first argument.
        seq = self.typed(N.Comma(tok.loc, args[1], args[0]))
        return new_cast(seq, T.TY_LONG)

    def _b_constant_p(self, tok):
        args = self._args(1)
        return N.IntLit(tok.loc, builtin_constant_p(args[0]), ty=T.TY_INT)

    def _inf_lit(self, tok, ty):
        """__builtin_huge_val*()/__builtin_inf*(): a constant +infinity."""
        return N.FloatLit(tok.loc, float("inf"), ty=ty)

    def _nan_lit(self, tok, ty):
        """__builtin_nan*(tag): a constant quiet NaN (the tag string only
        selects a payload, which is not modelled)."""
        self._args(1)
        return N.FloatLit(tok.loc, float("nan"), ty=ty)

    def _b_nan(self, tok):
        return self._nan_lit(tok, T.TY_DOUBLE)

    def _b_nanf(self, tok):
        return self._nan_lit(tok, T.TY_FLOAT)

    def _b_nanl(self, tok):
        return self._nan_lit(tok, T.TY_LDOUBLE)

    def _snan_lit(self, tok, ty):
        """__builtin_nans*(tag): a signaling NaN (quiet bit clear, payload 1). Only the
        double/float bit patterns are exact; long double is carried as a double."""
        self._args(1)
        if ty.kind == T.TypeKind.FLOAT:
            bits = 0x7F800001
            val = struct.unpack("<f", struct.pack("<I", bits))[0]
        else:
            bits = 0x7FF0000000000001
            val = struct.unpack("<d", struct.pack("<Q", bits))[0]
        return N.FloatLit(tok.loc, val, ty=ty, bits=bits)

    def _b_nans(self, tok):
        return self._snan_lit(tok, T.TY_DOUBLE)

    def _b_nansf(self, tok):
        return self._snan_lit(tok, T.TY_FLOAT)

    def _b_nansl(self, tok):
        return self._snan_lit(tok, T.TY_LDOUBLE)

    def _fp_class_call(self, tok, base):
        """__builtin_isnan/isinf/isfinite[f|l](x): a 0/1 predicate on x's bit pattern at
        its own width (codegen/fpclassify.py); the f/l spellings convert x first."""
        arg = self._args(1)[0]
        spelled = tok.text[len("__builtin_"):]
        sfx = "" if spelled in FP_CLASS_STEMS else spelled[-1]
        ty = {"f": T.TY_FLOAT, "l": T.TY_LDOUBLE}.get(sfx)
        operand = new_cast(arg, ty) if ty is not None else self._fp_value(arg)
        return self._call(tok, base, [operand], T.TY_INT)

    def _fpcmp(self, tok, name):
        """__builtin_is<op>(a, b): quiet comparison of two real floats; both
        operands are widened to long double if either is, else to double
        (exact for float). Codegen sets the result from ucomi flags."""
        args = self._args(2)
        wide = any(a.ty.kind == T.TypeKind.LDOUBLE for a in args)
        ty = T.TY_LDOUBLE if wide else T.TY_DOUBLE
        return self._call(tok, name, [new_cast(a, ty) for a in args], T.TY_INT)

    def _b_fpclassify(self, tok):
        """__builtin_fpclassify(nan, inf, normal, subnormal, zero, x): the five
        category constants must be integer constant expressions (glibc math.h)."""
        args = self._args(6)
        consts = []
        for a in args[:5]:
            v = try_eval_const(a)
            if v is None:
                self.error("__builtin_fpclassify: category arguments must be integer constants", tok)
            consts.append(N.IntLit(a.loc, v, ty=T.TY_INT))
        return self._call(tok, "fpclassify", consts + [self._fp_value(args[5])], T.TY_INT)

    def _b_isnormal(self, tok):
        return self._call(tok, "isnormal", [self._fp_value(self._args(1)[0])], T.TY_INT)

    def _b_isinf_sign(self, tok):
        return self._call(tok, "isinf_sign", [self._fp_value(self._args(1)[0])], T.TY_INT)

    def _fp_value(self, arg):
        """Floating operands keep their own width (classified bit-exactly);
        integer operands are converted to double like the C library macros do."""
        return arg if T.is_flonum(arg.ty) else new_cast(arg, T.TY_DOUBLE)

    def _b_isunordered(self, tok):
        return self._fpcmp(tok, "isunordered")

    def _b_isgreater(self, tok):
        return self._fpcmp(tok, "isgreater")

    def _b_isgreaterequal(self, tok):
        return self._fpcmp(tok, "isgreaterequal")

    def _b_isless(self, tok):
        return self._fpcmp(tok, "isless")

    def _b_islessequal(self, tok):
        return self._fpcmp(tok, "islessequal")

    def _b_islessgreater(self, tok):
        return self._fpcmp(tok, "islessgreater")

    def _b_isinf(self, tok):
        return self._fp_class_call(tok, "isinf")

    def _b_isinff(self, tok):
        return self._fp_class_call(tok, "isinf")

    def _b_isinfl(self, tok):
        return self._fp_class_call(tok, "isinf")

    def _b_isnan(self, tok):
        return self._fp_class_call(tok, "isnan")

    def _b_isnanf(self, tok):
        return self._fp_class_call(tok, "isnan")

    def _b_isnanl(self, tok):
        return self._fp_class_call(tok, "isnan")

    def _b_isfinite(self, tok):
        return self._fp_class_call(tok, "isfinite")

    def _b_isfinitef(self, tok):
        return self._fp_class_call(tok, "isfinite")

    def _b_isfinitel(self, tok):
        return self._fp_class_call(tok, "isfinite")

    def _b_isnormal(self, tok):
        """__builtin_isnormal(x) == (__fpclassify{,f,l}(x) == FP_NORMAL). glibc has
        no isnormal() symbol, so it is built on the libc fpclassify classifier.
        The operand is evaluated once into a temp. _Float16 has no libc classifier:
        it is widened exactly to double, and a value is normal there iff its
        magnitude is at least 2^-14 (the smallest normal _Float16)."""
        arg = self._args(1)[0]
        f16 = arg.ty.kind == T.TypeKind.FLOAT16
        sfx = {T.TypeKind.FLOAT: "f", T.TypeKind.LDOUBLE: "l"}.get(arg.ty.kind, "")
        cty = {"f": T.TY_FLOAT, "l": T.TY_LDOUBLE}.get(sfx, T.TY_DOUBLE)
        tmp = self.new_temp(cty, tok.loc)
        store = self.typed(N.Assign(tok.loc, self.typed(N.VarRef(tok.loc, tmp)),
                                    new_cast(arg, cty)))
        fn = self.implicit_function("__fpclassify" + sfx, tok, T.TY_INT)
        callee = self.typed(N.VarRef(tok.loc, fn))
        val = self.typed(N.VarRef(tok.loc, tmp))
        cls = self.typed(N.Call(tok.loc, callee, [val], fn.ty))
        res = self.typed(N.Binary(tok.loc, "==", cls, N.IntLit(tok.loc, 4, ty=T.TY_INT)))
        if f16:
            lim = 2.0 ** -14
            big = self.typed(N.Logical(tok.loc, "||",
                self.typed(N.Binary(tok.loc, ">=", val, N.FloatLit(tok.loc, lim, ty=T.TY_DOUBLE))),
                self.typed(N.Binary(tok.loc, "<=", self.typed(N.VarRef(tok.loc, tmp)),
                                    N.FloatLit(tok.loc, -lim, ty=T.TY_DOUBLE)))))
            res = self.typed(N.Logical(tok.loc, "&&", res, big))
        return self.typed(N.Comma(tok.loc, store, res))

    def _b_finite(self, tok):
        return self._fp_class_call(tok, "isfinite")

    def _b_finitef(self, tok):
        return self._fp_class_call(tok, "isfinite")

    def _b_finitel(self, tok):
        return self._fp_class_call(tok, "isfinite")

    def _b_huge_val(self, tok):
        return self._inf_lit(tok, T.TY_DOUBLE)

    def _b_inf(self, tok):
        return self._inf_lit(tok, T.TY_DOUBLE)

    def _b_huge_valf(self, tok):
        return self._inf_lit(tok, T.TY_FLOAT)

    def _b_inff(self, tok):
        return self._inf_lit(tok, T.TY_FLOAT)

    def _b_huge_vall(self, tok):
        return self._inf_lit(tok, T.TY_LDOUBLE)

    def _b_infl(self, tok):
        return self._inf_lit(tok, T.TY_LDOUBLE)

    def _b_signbit(self, tok):
        """__builtin_signbit(x): operand widened exactly to double (keeps the
        sign bit, also for NaN and float/long double); codegen reads bit 63."""
        args = [new_cast(a, T.TY_DOUBLE) for a in self._args(1)]
        return self._call(tok, "signbit", args, T.TY_INT)

    def _b_signbitf(self, tok):
        return self._b_signbit(tok)

    def _b_signbitl(self, tok):
        return self._b_signbit(tok)

    def _b_unreachable(self, tok):
        return self._call(tok, "trap", [], T.TY_VOID)

    def _b_trap(self, tok):
        return self._call(tok, "trap", [], T.TY_VOID)

    def _b_classify_type(self, tok):
        args = self._args(1)
        t = args[0].ty
        v = 1 if T.is_integer(t) else 8 if T.is_flonum(t) else 5 if t.base is not None else 12
        return N.IntLit(tok.loc, v, ty=T.TY_INT)

    def _b_types_compatible_p(self, tok):
        a = self.typename().unqual()
        self.expect(",")
        b = self.typename().unqual()
        return N.IntLit(tok.loc, int(is_compatible(a, b)), ty=T.TY_INT)

    def _b_choose_expr(self, tok):
        c = self.const_expr()
        self.expect(",")
        a = self.assign()
        self.expect(",")
        b = self.assign()
        return a if c else b

    def _b_offsetof(self, tok):
        ty = self.typename()
        self.expect(",")
        off = 0
        dyn = []      # run-time terms: array subscripts that are not integer constants
        cur = ty
        first = True
        while True:
            if not first and self.accept("->"):
                # GCC extension: `arr->m` in offsetof means `arr[0].m`
                if cur.kind == T.TypeKind.ARRAY:
                    cur = cur.base
                first = True
            if first or self.accept("."):
                first = False
                name = self.expect_ident()
                path = find_member(cur, name.text)
                if path is None:
                    self.error(f"'{cur}' has no member named '{name.text}'", name)
                for m in path:
                    off += m.offset
                    cur = m.ty
            elif self.accept("["):
                idx = self.conditional()
                self.expect("]")
                k = try_eval_const(idx)
                if k is not None:
                    off += k * cur.base.size
                else:       # GCC: a non-constant designator is evaluated at run time
                    term = N.Binary(tok.loc, "*", N.Cast(tok.loc, idx, ty=T.TY_LONG),
                                    N.IntLit(tok.loc, cur.base.size, ty=T.TY_LONG))
                    dyn.append(self.typed(term))
                cur = cur.base
            else:
                break
        if not dyn:
            return N.IntLit(tok.loc, off, ty=T.TY_ULONG)
        node = N.IntLit(tok.loc, off, ty=T.TY_LONG)
        for term in dyn:
            node = self.typed(N.Binary(tok.loc, "+", node, term))
        return self.typed(N.Cast(tok.loc, node, ty=T.TY_ULONG))
