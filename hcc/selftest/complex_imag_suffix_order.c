// EXPECT-EXIT: 0
// EXPECT-STDOUT: 0 1 0 2.5 0 4
// GNU imaginary constants with the i/j marker before a floating suffix (1.0iF, 2.5iL, 2.0if128).
#include <stdio.h>
int main(void) {
    _Complex float a = 1.0iF;
    _Complex double b = 2.5iL, c = 4.0i;
    printf("%g %g %g %g %g %g\n", __real__ a, __imag__ a, __real__ b, __imag__ b, __real__ c, __imag__ c);
    return 0;
}
