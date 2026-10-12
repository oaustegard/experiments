// EXPECT-EXIT: 0
// A signaling NaN literal keeps its bit pattern in a static float/double initializer
// (a Python float would quiet it), and __builtin_nans* uses GCC's encoding.
#include <string.h>
float fs = __builtin_nansf("");
double ds = -__builtin_nans("");
int main(void) {
    unsigned int u; unsigned long long v;
    float f = __builtin_nansf("");
    memcpy(&u, &fs, 4);
    if (u != 0x7fa00000u) return 1;
    memcpy(&v, &ds, 8);
    if (v != 0xfff4000000000000ull) return 2;
    if (memcmp(&f, &fs, 4) != 0) return 3;
    return 0;
}
