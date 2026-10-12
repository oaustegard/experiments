// EXPECT-EXIT: 0
// __builtin_<fn>f16: computed in float and rounded once to _Float16; constant
// initializers fold; integer-result forms keep their type.
extern void abort(void);
volatile _Float16 c1 = __builtin_sqrtf16(2.0f16);
volatile _Float16 c2 = __builtin_ldexpf16(1.5f16, 3);
volatile long c3 = __builtin_lroundf16(42.25f16);
int main(void) {
    volatile _Float16 r = __builtin_sqrtf16(c2 * 2);
    if (c1 != (_Float16)1.4140625f16) abort();    /* sqrt(2) rounded to half */
    if (c2 != 12.0f16) abort();
    if (c3 != 42) abort();
    if (r != (_Float16)4.8984375f16) abort();
    if (__builtin_ilogbf16(8.0f16) != 3) abort();
    return 0;
}
