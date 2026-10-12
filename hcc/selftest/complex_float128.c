// EXPECT-EXIT: 0
// EXPECT-STDOUT: 11 2
// EXPECT-STDOUT: 1
// EXPECT-STDOUT: 0
// _Complex _Float128: parts use the x87-precision path of complex long double; imaginary f128 constants.
#include <stdio.h>
int main(void) {
    _Complex _Float128 a = 1.0f128 + 2.0f128i, b = 3.0f128 - 4.0f128i, c;
    c = a * b;
    printf("%d %d\n", (int)__real__ c, (int)__imag__ c);
    printf("%d\n", (int)(0.5f128i * 2 == 1i));
    printf("%d\n", (int)__imag__ (_Complex _Float128) 2);
    return 0;
}
