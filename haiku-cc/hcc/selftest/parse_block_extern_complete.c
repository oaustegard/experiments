// EXPECT-EXIT: 0
// EXPECT-STDOUT: 2 12
#include <stdio.h>
int a[3] = { 1, 2, 3 };
int main(void) {
  int a;
  a = 5;
  { extern int a[3]; printf("%d %d\n", a[1], (int)sizeof a); }
  return a == 5 ? 0 : 1;
}
