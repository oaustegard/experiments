// EXPECT-STDOUT: 11 2
// EXPECT-STDOUT: -2 -4
// EXPECT-STDOUT: 0 1 0
#include <stdio.h>
_Complex double mul(_Complex double a, _Complex double b) { return a * b; }
_Complex double neg(_Complex double a) { return -a; }
int main(void) {
    _Complex double a = 1.0 + 2.0i, b = 3.0 - 4.0i;
    _Complex double c = mul(a, b);
    printf("%g %g\n", __real__ c, __imag__ c);
    _Complex double n = neg(a * 2.0);
    printf("%g %g\n", __real__ n, __imag__ n);
    printf("%d %d %d\n", a == b, a != b, (a - a) != 0.0);
    return 0;
}
