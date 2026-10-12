// EXPECT-EXIT: 0
// EXPECT-STDOUT: 100 200 1 2
// EXPECT-STDOUT: 1.5 -2.25 5 5
// EXPECT-STDOUT: 1 2 3 4
// File-scope _Complex initializers are folded to constant parts;
// a complex value as a condition tests both parts; complex struct members.
#include <stdio.h>
struct s { _Complex unsigned short x; };
struct s gs = { 100 + 200i };
_Complex char c1 = 1 + 2i;
_Complex float gf = 1.5f - 2.25i;
_Complex double gd = (1 + 2i) * (3 - 1i);
typedef struct { _Complex char a; _Complex char b; } Scc2;
Scc2 s2 = { 1+2i, 3+4i };
int main(void) {
    _Complex double zero = 0 + 0i, half = 0 + 0.5i;
    if (zero) return 1;
    if (!half) return 2;
    printf("%d %d %d %d\n", __real__ gs.x, __imag__ gs.x, (int)__real__ c1, (int)__imag__ c1);
    printf("%g %g %g %g\n", __real__ gf, __imag__ gf, __real__ gd, __imag__ gd);
    printf("%d %d %d %d\n", (int)__real__ s2.a, (int)__imag__ s2.a, (int)__real__ s2.b, (int)__imag__ s2.b);
    return 0;
}
