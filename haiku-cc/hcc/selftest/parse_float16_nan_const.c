// EXPECT-EXIT: 0
// _Float16 constants: NaN payloads survive (sNaN raises invalid on use), and
// constant folding rounds to binary16 before comparing.
#include <fenv.h>
extern void abort(void);
volatile _Float16 snan = __builtin_nansf16("");
volatile _Float16 qnan = __builtin_nanf16("");
int main(void) {
    volatile _Float16 r;
    if ((_Float16)2049.0 != 2048.0f16) abort();      /* folded at half precision */
    if (fetestexcept(FE_INVALID)) abort();
    r = qnan + qnan;
    if (fetestexcept(FE_INVALID)) abort();
    r = snan + snan;
    if (!fetestexcept(FE_INVALID)) abort();
    return 0;
}
