// EXPECT-STDOUT: -48
// EXPECT-STDOUT: 20 24 28 32
// GCC vectors of decimal lanes: per-lane libgcc calls, lane sign flip for unary minus.
#include <stdio.h>
typedef _Decimal64 __attribute__((__vector_size__ (8))) U;
typedef _Decimal64 __attribute__((__vector_size__ (16))) V;
typedef _Decimal32 __attribute__((__vector_size__ (16))) D4;
V v;
U foo(U u) { u *= u; u *= -(U){ v[1] }; return u; }
D4 bar(D4 a, D4 b) { return -(a - b) * b; }
int main(void) {
  v[0] = 2; v[1] = 3;
  U x = { 4 };
  U r = foo(x);
  printf("%d\n", (int)(r[0]));
  D4 a = {1, 2, 3, 4}, b = {5, 6, 7, 8};
  D4 c = bar(a, b);
  printf("%d %d %d %d\n", (int)c[0], (int)c[1], (int)c[2], (int)c[3]);
  return 0;
}
