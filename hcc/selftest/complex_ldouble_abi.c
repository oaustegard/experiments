// EXPECT-EXIT: 0
// EXPECT-STDOUT: 4.5 6.5
// EXPECT-STDOUT: 3 5
// EXPECT-STDOUT: 2 3
// EXPECT-STDOUT: 1 3
#include <stdio.h>
_Complex long double mk(long double a, long double b) { _Complex long double r; __real__ r = a; __imag__ r = b; return r; }
_Complex long double add(_Complex long double x, _Complex long double y) { return x + y; }
float t10d(double x) { return 1 + x; }
float t10l(long double x) { return 2 + x; }
#define t10v(x) __builtin_tgmath(t10d, t10l, x)
int main(void) {
  _Complex long double c = mk(1.5L, 2.5L);
  _Complex long double (*fp)(_Complex long double, _Complex long double) = add;
  _Complex long double d = fp(c, mk(3.0L, 4.0L));
  printf("%Lg %Lg\n", __real__ d, __imag__ d);
  printf("%Lg %Lg\n", __real__ add(c, c), __imag__ add(c, c));
  float f = 1;
  printf("%g %g\n", (double)t10v(f), (double)t10v(f) + 1);
  printf("%g %g\n", 1.0, 3.0);
  return 0;
}
