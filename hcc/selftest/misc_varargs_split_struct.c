// EXPECT-STDOUT: 1.0 3.0 13
// va_arg of structs whose eightbytes are SSE-class (struct {float a[3]}) or mixed (float + int) must come from the fp save area.
#include <stdarg.h>
#include <stdio.h>
struct F3 { float a[3]; };
struct FI { float f; int i; };
void get(int n, ...) {
  va_list ap; va_start(ap, n);
  struct F3 x = va_arg(ap, struct F3);
  struct FI y = va_arg(ap, struct FI);
  va_end(ap);
  printf("%.1f %.1f %d\n", x.a[0], x.a[2], y.i + 9 + n);
}
int main(void) {
  struct F3 s = {{1, 2, 3}};
  struct FI t = {4.0f, 4};
  get(0, s, t);
  return 0;
}
