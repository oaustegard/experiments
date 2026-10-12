// EXPECT-EXIT: 0
// EXPECT-STDOUT: i 0i 1 2
// A binary operand that contains a setjmp call must not have its sibling
// operand's temporary live across the call; longjmp re-enters with the
// correct left value (regression for the Looping.c SetjmpLongjmp test).
#include <stdio.h>
#include <setjmp.h>
int main(void) {
  jmp_buf buf;
  volatile int i = 0;
  int r = 10 + setjmp(buf);
  if (r < 12) {
    printf("i %d", i);
    i++;
    longjmp(buf, i);
  }
  printf(" %d\n", r - 10);
  return 0;
}
