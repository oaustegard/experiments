# hcc status

Measured with `python3 tests/run.py --suite <name> --jobs 4` at the commit
that last touched this file.

| suite | pass / total |
|---|---|
| c-testsuite | 213 / 220 |
| gcc-torture (execute) | 1442 / 1632 (66 skipped) |
| chibicc | 30 / 41 |
| hcc selftest | 10 / 10 |

## Works
Full preprocessor (macros incl. `#`/`##`/`__VA_ARGS__`/`__VA_OPT__`, GNU
comma elision, `#include_next`, include-guard caching, `__has_include`);
glibc headers parse; all integer types, pointers, arrays, structs/unions
(incl. bitfields, anonymous members, packed, flexible arrays), enums,
typedefs, initializers with designators, compound literals, statement
expressions, `?:` GNU form, `_Generic`, `_Static_assert`, float/double
arithmetic (SSE) and long double (x87), SysV calls incl. struct by value and
struct returns, variadic functions (`va_start/va_arg/va_copy`), switch with
GNU case ranges, computed goto (`&&label`), K&R function definitions,
`__attribute__` (aligned, packed, weak, section, visibility, alias, noreturn),
`__asm__` symbol labels, TLS variables, basic top-level/statement asm.

## TODO (good swarm tasks; each should come with a selftest)
* VLAs (`parser/locals.py`, `codegen` alloca of `vla_size`), sizeof on VLA
* `__int128` arithmetic (currently loads/stores low half only)
* `_Complex`: done for + - * / == != unary -, __real__/__imag__, imaginary constants, conversions; TODO: complex long double across calls (x87 ABI), `++`/`--`, complex in switch, `__divdc3`-exact division (inf/NaN)
* GNU vector extensions (`__attribute__((vector_size(N)))`)
* extended inline asm operands (`codegen/asm.py`)
* `__builtin_*`: mul/add/sub_overflow, bswap, clz/ctz/popcount, prefetch,
  return_address, signbit, isnan..., memcpy-style libc fallbacks already work
* va_arg of structs with SSE-class eightbytes (`codegen/varargs.py`)
* `#pragma pack`, `#line` (`preprocess/directives.py`)
* nested functions (gcc extension) — probably never
* alloca when temporaries are pushed (`codegen/builtins.py`)
* `-fPIC` code generation for shared libraries (GOT for extern data)
* x87 values across calls (long double binary ops with calls in rhs)
* cast of NaN in conditions (ucomis parity)
* _Float32 is a distinct float type (TY_FLOAT32, no default promotion). Caveat: with __GNUC__ 4.2, glibc's bits/floatn-common.h typedefs `_Float32` to float, so after <stdio.h>/<stdlib.h> it is plain float. gcc-dg/torture/float32-nan*.c (sNaN + fenv) and _Float32 imaginary suffixes (1.if32) are still open.
