#include <stdio.h>
// EXPECT-EXIT: 0
// EXPECT-STDOUT: 3 4
// EXPECT-STDOUT: 1
_Complex _Float16 h = 3.0f16 + 4.0f16 * 1.0if16;
int main(void) {
    _Complex _Float128 q = 1.0f128;
    printf("%g %g\n", (double)__real__ h, (double)__imag__ h);
    printf("%d\n", (int)(__real__ q == 1.0f128));
    return 0;
}
