// EXPECT-STDOUT: 6 4 1000000000
// EXPECT-STDOUT: 21
// EXPECT-STDOUT: 6 7
// EXPECT-STDOUT: 0
// _Decimal128 arguments beyond the xmm registers go on the stack (16 bytes each);
// struct members of decimal type; static initializers computed from constant expressions.
#include <stdio.h>
_Decimal64 g = 2 * 3;
_Decimal128 h = (1 ? 4 : 5);
_Decimal32 k = 1000000000L;
struct P { _Decimal64 a; _Decimal32 b; };
struct P mkp(_Decimal64 x) { struct P p; p.a = x; p.b = x + 1; return p; }
_Decimal128 many(_Decimal32 a, _Decimal64 b, _Decimal128 c, _Decimal128 d, _Decimal128 e, _Decimal128 f,
                 _Decimal128 g1, _Decimal128 h1, _Decimal128 i1, _Decimal128 j1, int n, _Decimal32 z) {
  return c + d + e + f + g1 + h1 + i1 + j1 + n * a + b + z;
}
int main(void) {
  _Decimal128 one = 1, two = 2;
  _Decimal128 r = many(1, 2, one, two, one, two, one, two, one, two, 3, 4);
  printf("%d %d %d\n", (int)g, (int)h, (int)k);
  printf("%d\n", (int)r);
  struct P p = mkp(g);
  printf("%d %d\n", (int)p.a, (int)p.b);
  _Decimal64 big = 123456789012345678LL;
  printf("%d\n", (int)(big - 123456789012345678LL));
  return 0;
}
