// EXPECT-EXIT: 0
// Ordered relations (< <= > >=) on a NaN raise FE_INVALID; == and != do not.
#include <fenv.h>
int main(void) {
    volatile double n = __builtin_nan(""), one = 1.0;
    volatile long double nl = __builtin_nanl(""), onel = 1.0L;
    volatile int r;
    feclearexcept(FE_ALL_EXCEPT);
    r = n == one; r = n != one; r = nl == onel; r = nl != onel;
    if (fetestexcept(FE_INVALID)) return 1;
    r = n < one; if (!fetestexcept(FE_INVALID)) return 2;
    feclearexcept(FE_ALL_EXCEPT);
    r = n >= one; if (!fetestexcept(FE_INVALID)) return 3;
    feclearexcept(FE_ALL_EXCEPT);
    r = nl <= onel; if (!fetestexcept(FE_INVALID)) return 4;
    feclearexcept(FE_ALL_EXCEPT);
    r = nl > onel; if (!fetestexcept(FE_INVALID)) return 5;
    return 0;
}
