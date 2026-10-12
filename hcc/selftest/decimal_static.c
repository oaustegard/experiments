// EXPECT-STDOUT: 1 3 6 6
// EXPECT-STDOUT: 10 20 9
// EXPECT-STDOUT: 10 9
// EXPECT-STDOUT: 1
// EXPECT-STDOUT: 4 16
// EXPECT-STDOUT: 16
// Static _Decimal objects initialised from integer constants get BID bits; locals,
// members, ++/--, the ?: operator and sizeof/_Alignof of the decimal types.
#include <stdio.h>
_Decimal64 g = 1;
_Decimal32 gs[2] = { 2, 3 };
struct S { int x; _Decimal128 d; _Decimal32 e; };
struct S sv = { 4, 5, 6 };
_Decimal64 pick(int k, _Decimal64 a) { return k ? a++ : --a; }
int main(void) {
  _Decimal64 x = 10;
  x++; ++x; x -= 2;
  struct S t;
  t.d = x; t.e = 9; t.x = 1;
  _Decimal128 y = t.d * 2;
  printf("%d %d %d %d\n", (int)g, (int)gs[1], (int)sv.e, (int)(sv.d + 1));
  printf("%d %d %d\n", (int)x, (int)y, (int)t.e);
  printf("%d %d\n", (int)pick(1, x), (int)pick(0, x));
  printf("%d\n", x > 5 ? 1 : 0);
  printf("%d %d\n", (int)sizeof(_Decimal32), (int)_Alignof(_Decimal128));
  _Decimal32 arr[3];
  arr[1] = 4;
  arr[2] = arr[1] * arr[1];
  printf("%d\n", (int)arr[2]);
  return 0;
}
