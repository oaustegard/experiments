// EXPECT-EXIT: 0
// EXPECT-STDOUT: resumed 1
// __builtin_setjmp returns 0 first and 1 after __builtin_longjmp to the same buf,
// including when the longjmp happens in a callee (frame and stack restored).
#include <stdio.h>
void *buf[5];
int g;
void jump(void) { __builtin_longjmp(buf, 1); }
int f(void) {
  int n = 0;
  if (__builtin_setjmp(buf)) return 100 + g;
  g = 7; n = 3; jump();
  return n;
}
int main(void) {
  int r = f();
  if (r != 107) return 1;
  if (__builtin_setjmp(buf) == 0) __builtin_longjmp(buf, 1);
  printf("resumed %d\n", 1);
  return 0;
}
