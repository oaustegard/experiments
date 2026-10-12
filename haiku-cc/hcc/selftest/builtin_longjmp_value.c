// EXPECT-EXIT: 0
// EXPECT-STDOUT: 6
// EXPECT-STDOUT: 1
// __builtin_setjmp returns the value given to __builtin_longjmp (0 becomes 1).
#include <stdio.h>
void *buf[5];
int main(void) {
  int r = 0;
  r = __builtin_setjmp(buf);
  if (r < 5) __builtin_longjmp(buf, r + 2);
  printf("%d\n", r);
  int z = 0;
  z = __builtin_setjmp(buf);
  if (z == 0) __builtin_longjmp(buf, 0);
  printf("%d\n", z);
  return 0;
}
