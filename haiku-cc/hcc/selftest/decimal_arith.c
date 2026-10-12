// EXPECT-STDOUT: 45
// EXPECT-STDOUT: 33
// EXPECT-STDOUT: -12
// EXPECT-STDOUT: 12 9 6
// EXPECT-STDOUT: 22 1
// EXPECT-STDOUT: 12345678901
// EXPECT-STDOUT: -5 200
// EXPECT-STDOUT: true
// EXPECT-STDOUT: 1
// _Decimal32/64/128: BID arithmetic, comparisons, unary minus and conversions via libgcc.
#include <stdio.h>
_Decimal64 mul(_Decimal64 x, _Decimal64 y) { return x * y + 3; }
_Decimal128 wide(_Decimal128 a, int i, _Decimal32 s) { return a * i - s; }
_Decimal32 small(char c, _Decimal32 d) { d *= c; d = -d; return d; }
int cmp(_Decimal64 a, _Decimal64 b) {
  return (a < b) * 1 + (a == b) * 2 + (a >= b) * 4 + (a != b) * 8 + !a * 16;
}
int main(void) {
  _Decimal64 a = 7, b = 6;
  printf("%d\n", (int)mul(a, b));
  printf("%d\n", (int)wide(a, 5, 2));
  printf("%d\n", (int)small(4, 3));
  printf("%d %d %d\n", cmp(a, b), cmp(b, a), cmp(a, a));
  _Decimal64 z = 0;
  printf("%d %d\n", cmp(z, z), (int)(-z == 0));
  unsigned long u = 12345678901234ul;
  _Decimal64 big = u;
  printf("%ld\n", (long)(big / 1000));
  long neg = -5;
  _Decimal128 nd = neg;
  unsigned char uc = 200;
  _Decimal32 ud = uc;
  printf("%ld %d\n", (long)nd, (int)ud);
  if (a) puts("true");
  _Bool bb = a;
  printf("%d\n", bb);
  return 0;
}
