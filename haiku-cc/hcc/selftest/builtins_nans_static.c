// EXPECT-EXIT: 0
// __builtin_nans is a signaling NaN: it is a constant usable in static
// initializers, and its quiet bit is clear.
#include <string.h>
volatile double d = __builtin_nans("");
float f = __builtin_nansf("1");
int main(void) {
    unsigned long long bits;
    memcpy(&bits, &d, sizeof bits);
    if ((bits >> 51 & 1) != 0 || (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL) return 1;
    if (f == f) return 2;
    return 0;
}
