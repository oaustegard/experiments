"""Predefined macros (gcc x86-64 LP64 compatible) and dynamic builtin macros.

`predefined_source(opts)` returns C source text of `#define` lines that the
Preprocessor processes before the main file. Dynamic macros (__FILE__,
__LINE__, __COUNTER__, ...) are implemented by `DYNAMIC_MACROS`, functions
(pp, tok) -> list[Token].
"""
from __future__ import annotations

import os
import time

from hcc.preprocess.floatn_macros import FLOATN_MACROS
from hcc.tokens import Token, TokenKind

PREDEFINED = r"""
#define __STDC__ 1
#define __STDC_HOSTED__ 1
#define __STDC_UTF_16__ 1
#define __STDC_UTF_32__ 1
#define __STDC_IEC_559__ 1
#define __STDC_IEC_559_COMPLEX__ 1
#define __STDC_ISO_10646__ 201706L
#define __STDC_NO_ATOMICS__ 1
#define __STDC_NO_COMPLEX__ 1
#define __STDC_NO_THREADS__ 1
#define __STDC_NO_VLA__ 1
#define __hcc__ 1
#define __GNUC__ 4
#define __GNUC_MINOR__ 2
#define __GNUC_PATCHLEVEL__ 1
#define __VERSION__ "hcc 0.1 (gcc 4.2 compatible)"
#define __x86_64__ 1
#define __x86_64 1
#define __amd64__ 1
#define __amd64 1
#define __LP64__ 1
#define _LP64 1
#define __linux__ 1
#define __linux 1
#define __gnu_linux__ 1
#define __unix__ 1
#define __unix 1
#define linux 1
#define unix 1
#define __ELF__ 1
#define __CHAR_BIT__ 8
#define __SIZEOF_SHORT__ 2
#define __SIZEOF_INT__ 4
#define __SIZEOF_LONG__ 8
#define __SIZEOF_LONG_LONG__ 8
#define __SIZEOF_POINTER__ 8
#define __SIZEOF_FLOAT__ 4
#define __SIZEOF_DOUBLE__ 8
#define __SIZEOF_LONG_DOUBLE__ 16
#define __SIZEOF_SIZE_T__ 8
#define __SIZEOF_WCHAR_T__ 4
#define __SIZEOF_WINT_T__ 4
#define __SIZEOF_PTRDIFF_T__ 8
#define __SIZEOF_INT128__ 16
#define __BIGGEST_ALIGNMENT__ 16
#define __SIZE_TYPE__ unsigned long
#define __PTRDIFF_TYPE__ long
#define __WCHAR_TYPE__ int
#define __WINT_TYPE__ unsigned int
#define __INTMAX_TYPE__ long
#define __UINTMAX_TYPE__ unsigned long
#define __INTPTR_TYPE__ long
#define __UINTPTR_TYPE__ unsigned long
#define __CHAR16_TYPE__ unsigned short
#define __CHAR32_TYPE__ unsigned int
#define __INT8_TYPE__ signed char
#define __INT16_TYPE__ short
#define __INT32_TYPE__ int
#define __INT64_TYPE__ long
#define __UINT8_TYPE__ unsigned char
#define __UINT16_TYPE__ unsigned short
#define __UINT32_TYPE__ unsigned int
#define __UINT64_TYPE__ unsigned long
#define __SCHAR_MAX__ 0x7f
#define __SHRT_MAX__ 0x7fff
#define __INT_MAX__ 0x7fffffff
#define __LONG_MAX__ 0x7fffffffffffffffL
#define __LONG_LONG_MAX__ 0x7fffffffffffffffLL
#define __WCHAR_MAX__ 0x7fffffff
#define __WCHAR_MIN__ (-__WCHAR_MAX__ - 1)
#define __WINT_MIN__ 0U
#define __WINT_MAX__ 0xffffffffU
#define __WINT_WIDTH__ 32
#define __SIZE_MAX__ 0xffffffffffffffffUL
#define __PTRDIFF_MAX__ 0x7fffffffffffffffL
#define __INTMAX_MAX__ 0x7fffffffffffffffL
#define __UINTMAX_MAX__ 0xffffffffffffffffUL
#define __INTPTR_MAX__ 0x7fffffffffffffffL
#define __UINTPTR_MAX__ 0xffffffffffffffffUL
#define __SIG_ATOMIC_TYPE__ int
#define __SIG_ATOMIC_MAX__ 0x7fffffff
#define __SIG_ATOMIC_MIN__ (-__SIG_ATOMIC_MAX__ - 1)
#define __SIG_ATOMIC_WIDTH__ 32
#define __SCHAR_WIDTH__ 8
#define __SHRT_WIDTH__ 16
#define __INT_WIDTH__ 32
#define __INTPTR_WIDTH__ 64
#define __WCHAR_WIDTH__ 32
#define __LONG_WIDTH__ 64
#define __LONG_LONG_WIDTH__ 64
#define __SIZE_WIDTH__ 64
#define __PTRDIFF_WIDTH__ 64
#define __INTMAX_WIDTH__ 64
#define __INT_LEAST8_WIDTH__ 8
#define __INT_LEAST16_WIDTH__ 16
#define __INT_LEAST32_WIDTH__ 32
#define __INT_LEAST64_WIDTH__ 64
#define __INT_FAST8_WIDTH__ 8
#define __INT_FAST16_WIDTH__ 64
#define __INT_FAST32_WIDTH__ 64
#define __INT_FAST64_WIDTH__ 64
#define __INT8_MAX__ 0x7f
#define __INT16_MAX__ 0x7fff
#define __INT32_MAX__ 0x7fffffff
#define __INT64_MAX__ 0x7fffffffffffffffL
#define __UINT8_MAX__ 0xff
#define __UINT16_MAX__ 0xffff
#define __UINT32_MAX__ 0xffffffffU
#define __UINT64_MAX__ 0xffffffffffffffffUL
#define __INT_LEAST8_TYPE__ signed char
#define __INT_LEAST16_TYPE__ short
#define __INT_LEAST32_TYPE__ int
#define __INT_LEAST64_TYPE__ long
#define __UINT_LEAST8_TYPE__ unsigned char
#define __UINT_LEAST16_TYPE__ unsigned short
#define __UINT_LEAST32_TYPE__ unsigned int
#define __UINT_LEAST64_TYPE__ unsigned long
#define __INT_FAST8_TYPE__ signed char
#define __INT_FAST16_TYPE__ long
#define __INT_FAST32_TYPE__ long
#define __INT_FAST64_TYPE__ long
#define __UINT_FAST8_TYPE__ unsigned char
#define __UINT_FAST16_TYPE__ unsigned long
#define __UINT_FAST32_TYPE__ unsigned long
#define __UINT_FAST64_TYPE__ unsigned long
#define __INT_LEAST8_MAX__ 0x7f
#define __INT_LEAST16_MAX__ 0x7fff
#define __INT_LEAST32_MAX__ 0x7fffffff
#define __INT_LEAST64_MAX__ 0x7fffffffffffffffL
#define __UINT_LEAST8_MAX__ 0xff
#define __UINT_LEAST16_MAX__ 0xffff
#define __UINT_LEAST32_MAX__ 0xffffffffU
#define __UINT_LEAST64_MAX__ 0xffffffffffffffffUL
#define __INT_FAST8_MAX__ 0x7f
#define __INT_FAST16_MAX__ 0x7fffffffffffffffL
#define __INT_FAST32_MAX__ 0x7fffffffffffffffL
#define __INT_FAST64_MAX__ 0x7fffffffffffffffL
#define __UINT_FAST8_MAX__ 0xff
#define __UINT_FAST16_MAX__ 0xffffffffffffffffUL
#define __UINT_FAST32_MAX__ 0xffffffffffffffffUL
#define __UINT_FAST64_MAX__ 0xffffffffffffffffUL
#define __INT8_C(c) c
#define __INT16_C(c) c
#define __INT32_C(c) c
#define __INT64_C(c) c ## L
#define __UINT8_C(c) c
#define __UINT16_C(c) c
#define __UINT32_C(c) c ## U
#define __UINT64_C(c) c ## UL
#define __INTMAX_C(c) c ## L
#define __UINTMAX_C(c) c ## UL
#define __FLT_MIN__ 1.17549435082228750796873653722224568e-38F
#define __FLT_MAX__ 3.40282346638528859811704183484516925e+38F
#define __FLT_EPSILON__ 1.19209289550781250000000000000000000e-7F
#define __DBL_MIN__ ((double)2.22507385850720138309023271733240406e-308L)
#define __DBL_MAX__ ((double)1.79769313486231570814527423731704357e+308L)
#define __DBL_EPSILON__ ((double)2.22044604925031308084726333618164062e-16L)
#define __FLT_MANT_DIG__ 24
#define __DBL_MANT_DIG__ 53
#define __LDBL_MANT_DIG__ 64
#define __FLT_DIG__ 6
#define __DBL_DIG__ 15
#define __FLT_RADIX__ 2
#define __DBL_MIN_EXP__ (-1021)
#define __FLT_EVAL_METHOD__ 0
#define __DBL_MIN_10_EXP__ (-307)
#define __LDBL_MAX__ 1.18973149535723176502126385303097021e+4932L
#define __LDBL_IS_IEC_60559__ 1
#define __FLT_EVAL_METHOD_TS_18661_3__ 0
#define __DBL_DENORM_MIN__ ((double)4.94065645841246544176568792868221372e-324L)
#define __LDBL_MIN__ 3.36210314311209350626267781732175260e-4932L
#define __DBL_IS_IEC_60559__ 1
#define __LDBL_HAS_QUIET_NAN__ 1
#define __FLT_HAS_DENORM__ 1
#define __DBL_HAS_INFINITY__ 1
#define __LDBL_HAS_DENORM__ 1
#define __DBL_MAX_EXP__ 1024
#define __FLT_MIN_EXP__ (-125)
#define __DBL_HAS_QUIET_NAN__ 1
#define __LDBL_MAX_EXP__ 16384
#define __DBL_HAS_DENORM__ 1
#define __LDBL_DECIMAL_DIG__ 21
#define __FLT_IS_IEC_60559__ 1
#define __LDBL_EPSILON__ 1.08420217248550443400745280086994171e-19L
#define __LDBL_DIG__ 18
#define __FLT_HAS_QUIET_NAN__ 1
#define __FLT_MAX_10_EXP__ 38
#define __FLT_HAS_INFINITY__ 1
#define __LDBL_HAS_INFINITY__ 1
#define __FLT_NORM_MAX__ 3.40282346638528859811704183484516925e+38F
#define __FLT_MAX_EXP__ 128
#define __LDBL_MIN_EXP__ (-16381)
#define __LDBL_MAX_10_EXP__ 4932
#define __DBL_DECIMAL_DIG__ 17
#define __DBL_NORM_MAX__ ((double)1.79769313486231570814527423731704357e+308L)
#define __FLT_DENORM_MIN__ 1.40129846432481707092372958328991613e-45F
#define __FLT_MIN_10_EXP__ (-37)
#define __DBL_MAX_10_EXP__ 308
#define __LDBL_DENORM_MIN__ 3.64519953188247460252840593361941982e-4951L
#define __LDBL_MIN_10_EXP__ (-4931)
#define __FLT_DECIMAL_DIG__ 9
#define __LDBL_NORM_MAX__ 1.18973149535723176502126385303097021e+4932L
#define __ORDER_LITTLE_ENDIAN__ 1234
#define __ORDER_BIG_ENDIAN__ 4321
#define __ORDER_PDP_ENDIAN__ 3412
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#define __FLOAT_WORD_ORDER__ __ORDER_LITTLE_ENDIAN__
#define __USER_LABEL_PREFIX__
#define __REGISTER_PREFIX__
#define __NO_INLINE__ 1
#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5
#define __SCHAR_WIDTH__ 8
#define __SHRT_WIDTH__ 16
#define __INT_WIDTH__ 32
#define __LONG_WIDTH__ 64
#define __LONG_LONG_WIDTH__ 64
#define __WCHAR_WIDTH__ 32
#define __PTRDIFF_WIDTH__ 64
#define __SIZE_WIDTH__ 64
#define __INTMAX_WIDTH__ 64
#define __INTPTR_WIDTH__ 64
#define __INT_LEAST8_WIDTH__ 8
#define __INT_LEAST16_WIDTH__ 16
#define __INT_LEAST32_WIDTH__ 32
#define __INT_LEAST64_WIDTH__ 64
#define __INT_FAST8_WIDTH__ 8
#define __INT_FAST16_WIDTH__ 64
#define __INT_FAST32_WIDTH__ 64
#define __INT_FAST64_WIDTH__ 64
#define __GCC_ATOMIC_BOOL_LOCK_FREE 2
#define __GCC_ATOMIC_CHAR_LOCK_FREE 2
#define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
#define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
#define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
#define __GCC_ATOMIC_SHORT_LOCK_FREE 2
#define __GCC_ATOMIC_INT_LOCK_FREE 2
#define __GCC_ATOMIC_LONG_LOCK_FREE 2
#define __GCC_ATOMIC_LLONG_LOCK_FREE 2
#define __GCC_ATOMIC_POINTER_LOCK_FREE 2
#define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
#define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
#define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
#define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
#define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
#define __alignof__ _Alignof
#define __builtin_va_end(ap) ((void)(ap))
#define __FLT16_HAS_QUIET_NAN__ 1
#define __FLT128_MAX_10_EXP__ 4932
#define __FLT32_MIN_EXP__ (-125)
#define __FLT32X_DECIMAL_DIG__ 17
#define __FLT64_DECIMAL_DIG__ 17
#define __FLT32X_MAX_EXP__ 1024
#define __FLT32_HAS_DENORM__ 1
#define __FLT32_MAX_10_EXP__ 38
#define __FLT64X_MAX_10_EXP__ 4932
#define __FLT64X_HAS_QUIET_NAN__ 1
#define __FLT128_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966F128
#define __FLT128_MIN_EXP__ (-16381)
#define __FLT128_MIN_10_EXP__ (-4931)
#define __FLT32X_IS_IEC_60559__ 1
#define __FLT128_MANT_DIG__ 113
#define __FLT32X_MANT_DIG__ 53
#define __FLT64X_EPSILON__ 1.08420217248550443400745280086994171e-19F64x
#define __FLT32_IS_IEC_60559__ 1
#define __FLT64X_DENORM_MIN__ 3.64519953188247460252840593361941982e-4951F64x
#define __FLT32X_HAS_INFINITY__ 1
#define __FLT64_EPSILON__ 2.22044604925031308084726333618164062e-16F64
#define __FLT128_IS_IEC_60559__ 1
#define __FLT64X_MIN_10_EXP__ (-4931)
#define __FLT16_MIN_EXP__ (-13)
#define __FLT64_MANT_DIG__ 53
#define __FLT64X_MANT_DIG__ 64
#define __FLT64_MAX_10_EXP__ 308
#define __FLT16_MAX_10_EXP__ 4
#define __FLT32X_HAS_DENORM__ 1
#define __FLT128_HAS_INFINITY__ 1
#define __FLT32_MAX__ 3.40282346638528859811704183484516925e+38F32
#define __FLT16_DECIMAL_DIG__ 5
#define __FLT64X_MIN_EXP__ (-16381)
#define __FLT32_MAX_EXP__ 128
#define __FLT64_DENORM_MIN__ 4.94065645841246544176568792868221372e-324F64
#define __FLT16_DENORM_MIN__ 5.96046447753906250000000000000000000e-8F16
#define __FLT128_EPSILON__ 1.92592994438723585305597794258492732e-34F128
#define __FLT64X_NORM_MAX__ 1.18973149535723176502126385303097021e+4932F64x
#define __FLT32X_EPSILON__ 2.22044604925031308084726333618164062e-16F32x
#define __FLT64_MIN_EXP__ (-1021)
#define __FLT64_MIN_10_EXP__ (-307)
#define __FLT16_MIN_10_EXP__ (-4)
#define __FLT64X_DECIMAL_DIG__ 21
#define __FLT32_MIN__ 1.17549435082228750796873653722224568e-38F32
#define __FLT128_MAX_EXP__ 16384
#define __FLT32_MANT_DIG__ 24
#define __FLT32X_MIN_EXP__ (-1021)
#define __FLT128_HAS_DENORM__ 1
#define __FLT32_DECIMAL_DIG__ 9
#define __FLT128_DIG__ 33
#define __FLT64X_DIG__ 18
#define __FLT16_DIG__ 3
#define __FLT32X_MIN__ 2.22507385850720138309023271733240406e-308F32x
#define __FLT64X_HAS_INFINITY__ 1
#define __FLT32_HAS_INFINITY__ 1
#define __FLT64_IS_IEC_60559__ 1
#define __FLT16_IS_IEC_60559__ 1
#define __FLT16_MAX_EXP__ 16
#define __FLT64_DIG__ 15
#define __FLT16_EPSILON__ 9.76562500000000000000000000000000000e-4F16
#define __FLT64X_HAS_DENORM__ 1
#define __FLT16_NORM_MAX__ 6.55040000000000000000000000000000000e+4F16
#define __FLT32_DENORM_MIN__ 1.40129846432481707092372958328991613e-45F32
#define __FLT16_MAX__ 6.55040000000000000000000000000000000e+4F16
#define __FLT128_MIN__ 3.36210314311209350626267781732175260e-4932F128
#define __FLT128_MAX__ 1.18973149535723176508575932662800702e+4932F128
#define __FLT64_HAS_QUIET_NAN__ 1
#define __FLT32_MIN_10_EXP__ (-37)
#define __FLT32X_DIG__ 15
#define __FLT64_HAS_INFINITY__ 1
#define __FLT64X_MAX__ 1.18973149535723176502126385303097021e+4932F64x
#define __FLT16_HAS_INFINITY__ 1
#define __FLT16_MANT_DIG__ 11
#define __FLT64X_MAX_EXP__ 16384
#define __FLT32_NORM_MAX__ 3.40282346638528859811704183484516925e+38F32
#define __FLT64_MAX__ 1.79769313486231570814527423731704357e+308F64
#define __FLT32X_MAX_10_EXP__ 308
#define __FLT16_MIN__ 6.10351562500000000000000000000000000e-5F16
#define __FLT64X_IS_IEC_60559__ 1
#define __FLT64_MAX_EXP__ 1024
#define __FLT128_NORM_MAX__ 1.18973149535723176508575932662800702e+4932F128
#define __FLT64_NORM_MAX__ 1.79769313486231570814527423731704357e+308F64
#define __FLT128_HAS_QUIET_NAN__ 1
#define __FLT64X_MIN__ 3.36210314311209350626267781732175260e-4932F64x
#define __FLT64_HAS_DENORM__ 1
#define __FLT32_EPSILON__ 1.19209289550781250000000000000000000e-7F32
#define __FLT16_HAS_DENORM__ 1
#define __FLT32X_MAX__ 1.79769313486231570814527423731704357e+308F32x
#define __FLT32_DIG__ 6
#define __FLT32X_NORM_MAX__ 1.79769313486231570814527423731704357e+308F32x
#define __FLT64_MIN__ 2.22507385850720138309023271733240406e-308F64
#define __FLT32X_HAS_QUIET_NAN__ 1
#define __FLT32X_DENORM_MIN__ 4.94065645841246544176568792868221372e-324F32x
#define __FLT32X_MIN_10_EXP__ (-307)
#define __FLT128_DECIMAL_DIG__ 36
#define __FLT32_HAS_QUIET_NAN__ 1
"""

STD_VERSIONS = {"c89": "199409L", "c90": "199409L", "gnu89": "199409L", "gnu90": "199409L",
                "c99": "199901L", "gnu99": "199901L", "c11": "201112L", "gnu11": "201112L",
                "c17": "201710L", "c18": "201710L", "gnu17": "201710L", "gnu18": "201710L",
                "c2x": "202000L", "gnu2x": "202000L", "c23": "202311L", "gnu23": "202311L"}


# GCC spellings of the ISO standard names (-std=iso9899:2017 etc.).
STD_ALIASES = {"iso9899:1990": "c90", "iso9899:199409": "c89", "iso9899:1999": "c99",
               "iso9899:2011": "c11", "iso9899:2017": "c17", "iso9899:2018": "c17",
               "c1x": "c11", "gnu1x": "gnu11"}


def predefined_source(opts) -> str:
    std = (opts.std or "gnu17").lower()
    std = STD_ALIASES.get(std, std)
    lines = [PREDEFINED, FLOATN_MACROS, f"#define __STDC_VERSION__ {STD_VERSIONS.get(std, '201710L')}"]
    if std.startswith("c"):
        lines.append("#define __STRICT_ANSI__ 1")
    # gnu89 inline semantics (-fgnu89-inline or C89/gnu89 modes) reflect
    # __GNUC_GNU_INLINE__; otherwise C99 inline semantics: __GNUC_STDC_INLINE__.
    if opts.gnu89_inline or std in ("c89", "c90", "gnu89", "gnu90"):
        lines.append("#define __GNUC_GNU_INLINE__ 1")
    else:
        lines.append("#define __GNUC_STDC_INLINE__ 1")
    if STD_VERSIONS.get(std, "") >= "202000L":
        # C23 makes static_assert a keyword (C11/C17 provide it via <assert.h>)
        lines.append("#define static_assert _Static_assert")
    if opts.pic:
        lines.append("#define __PIC__ 2\n#define __pic__ 2")
    return "\n".join(lines) + "\n"


def _num(tok: Token, n: int) -> list[Token]:
    t = Token(TokenKind.NUMBER, str(n), tok.loc, tok.bol, tok.space)
    return [t]


def _str(tok: Token, s: str) -> list[Token]:
    esc = s.replace("\\", "\\\\").replace('"', '\\"')
    return [Token(TokenKind.STRING, f'"{esc}"', tok.loc, tok.bol, tok.space)]


def _file(pp, tok):
    return _str(tok, pp.presumed(tok.loc)[0])


def _line(pp, tok):
    return _num(tok, pp.presumed(tok.loc)[1])


def _counter(pp, tok):
    pp.counter += 1
    return _num(tok, pp.counter - 1)


def _base_file(pp, tok):
    return _str(tok, pp.base_file)


_NOW = time.localtime()


def _date(pp, tok):
    return _str(tok, time.strftime("%b %e %Y", _NOW))


def _time(pp, tok):
    return _str(tok, time.strftime("%H:%M:%S", _NOW))


def _timestamp(pp, tok):
    # gcc: modification time of the current source file, ctime(3) layout
    # "Www Mmm dd hh:mm:ss yyyy" (always 24 chars, day space-padded).
    try:
        tm = time.localtime(os.stat(tok.loc.file).st_mtime)
    except OSError:
        tm = _NOW
    day = f"{tm.tm_mday:2d}"
    return _str(tok, time.strftime("%a %b ", tm) + day
                + time.strftime(" %H:%M:%S %Y", tm))


DYNAMIC_MACROS = {
    "__FILE__": _file, "__LINE__": _line, "__COUNTER__": _counter,
    "__BASE_FILE__": _base_file, "__DATE__": _date, "__TIME__": _time,
    "__TIMESTAMP__": _timestamp, "__INCLUDE_LEVEL__": lambda pp, tok: _num(tok, 0),
}
