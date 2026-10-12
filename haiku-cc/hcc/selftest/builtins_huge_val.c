// EXPECT-EXIT: 0
#include <math.h>
int main(void) {
    double d = __builtin_huge_val();
    float f = __builtin_inff();
    long double l = __builtin_infl();
    if (!(d > 1e308) || !(f > 1e38f) || !(l > 1e308L)) return 1;
    if (HUGE_VAL != d) return 2;
    return 0;
}
