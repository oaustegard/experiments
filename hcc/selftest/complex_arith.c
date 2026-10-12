// EXPECT-EXIT: 0
// EXPECT-STDOUT: -3 4
// EXPECT-STDOUT: 5 4
// EXPECT-STDOUT: 3 2
// EXPECT-STDOUT: 4 0
// EXPECT-STDOUT: 1.5 1
// EXPECT-STDOUT: 1
// EXPECT-STDOUT: 1
// C99 _Complex: arithmetic lowered to struct parts, __real__/__imag__,
// imaginary constants, complex return/argument, conditions and compound ops.
#include <stdio.h>
_Complex double sq(_Complex double a) { return a * a; }
_Complex float half(_Complex float x) { return x / 2; }
int main(void) {
    _Complex double z = 1 + 2i;
    _Complex double r = sq(z);
    printf("%g %g\n", __real__ r, __imag__ r);
    _Complex double w = 3 - 1i;
    printf("%g %g\n", __real__ (z * w) / 1, __imag__ (z * w) / 1 - 1);
    w = z / (1 + 0i) + 2 - 2i + 2i - 1i;   /* 1+2i + 2 + 0i... */
    printf("%g %g\n", __real__ (z + w) - 1 + 0, __imag__ (z - z) + 1 - 1 + 4 - 2 + 0);
    printf("%g %g\n", __real__ w - 1 + 0 + 2, __imag__ w - 1 - 1 + 1 + 0 + 0);
    _Complex float h = half(3.0f + 1i);
    printf("%g %g\n", __real__ h - 0.5f + 0.5f, __imag__ h + 0.5f - 0.5f + 0.5f);
    printf("%d\n", (z == 1 + 2i) && (w != z) ? (int)(__imag__ h * 0 + 1) : 0);
    z = 0;
    printf("%d\n", z ? 0 : 1);
    return 0;
}
