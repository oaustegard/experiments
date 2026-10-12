// EXPECT-STDOUT: 2 7 7 9
// Conversions between decimal and binary floating types and between decimal widths.
#include <stdio.h>
_Decimal64 a(double x) { return x; }
_Decimal32 b(float x) { return x * 2; }
double c(_Decimal128 d) { return d; }
float e(_Decimal64 d) { return d; }
int main(void) {
  printf("%d %d %g %g\n", (int)a(2.5), (int)b(3.75f), c(7), (double)e(9));
  return 0;
}
