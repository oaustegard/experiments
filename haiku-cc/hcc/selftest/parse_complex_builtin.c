// EXPECT-EXIT: 0
// __builtin_complex(re, im) as a file-scope constant initializer, a block-scope value
// and with a non-constant part (block scope only).
#include <stdio.h>
_Complex double g = __builtin_complex (1.5, -2.0);
_Complex float gf = __builtin_complex ((float) 1.0, 3.0f / 2.0f);
int main (void) {
    double x = 4.0;
    _Complex double c = __builtin_complex (x, 0.5);
    int bad = 0;
    if (__real__ g != 1.5 || __imag__ g != -2.0) bad = 1;
    if (__real__ gf != 1.0f || __imag__ gf != 1.5f) bad = 1;
    if (__real__ c != 4.0 || __imag__ c != 0.5) bad = 1;
    return bad;
}
