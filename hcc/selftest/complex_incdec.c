// EXPECT-EXIT: 0
// EXPECT-STDOUT: 2 2
// EXPECT-STDOUT: 3 2 | 2 2
// EXPECT-STDOUT: 2 2
// C23: ++/-- on a _Complex operand step the real part by 1; post form yields old value.
#include <stdio.h>
int main(void) {
    _Complex float a = 1.0f + 2.0fi;
    ++a;
    printf("%g %g\n", __real__ a, __imag__ a);
    _Complex float b = a++;
    printf("%g %g | %g %g\n", __real__ a, __imag__ a, __real__ b, __imag__ b);
    --a;
    printf("%g %g\n", __real__ a, __imag__ a);
    return 0;
}
