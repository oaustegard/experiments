"""_Float16 (IEEE binary16) support for the x86-64 backend.

Register model: a _Float16 value lives in %xmm0 as a single-precision float
whose value is exactly representable in binary16. Every producer keeps that
invariant by rounding to half (ROUND_HALF) after arithmetic and conversions;
memory holds the 2-byte half encoding, converted by LOAD_HALF / STORE_HALF.
These use F16C (vcvtph2ps / vcvtps2ph), which every x86-64 CPU since 2012 has.
Parameters and returns use the SysV ABI encoding: the half bits sit in the
low 16 bits of an %xmm register (see calls.py / functions.py).
"""

# %xmm0 (float value) -> nearest-even half -> back to float; value unchanged.
ROUND_HALF = "vcvtps2ph $0, %xmm0, %xmm0; vcvtph2ps %xmm0, %xmm0"
# *(%rax) (2-byte half) -> float in %xmm0
LOAD_HALF = "movzwl (%rax), %eax; vmovd %eax, %xmm0; vcvtph2ps %xmm0, %xmm0"
# %xmm0 (float value) -> 2 bytes at (%rdi); %xmm0 is preserved
STORE_HALF = "vcvtps2ph $0, %xmm0, %xmm1; vmovd %xmm1, %eax; mov %ax, (%rdi)"
# %xmm0 (float value) -> half bit pattern in the low 16 bits of %xmm0 (ABI)
TO_ABI_HALF = "vcvtps2ph $0, %xmm0, %xmm0"
# low 16 bits of %xmm<n> (ABI half) -> float value in %xmm0
FROM_ABI_HALF = "vcvtph2ps %xmm0, %xmm0"
