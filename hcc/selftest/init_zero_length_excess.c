// EXPECT-EXIT: 0
// EXPECT-STDOUT: 2 4 5
// Excess brace-elided initializers for zero-length arrays and members are
// skipped (GCC: "excess elements in initializer"), not a parse error.
struct s0 { };
struct s2 { int z[0]; int b; };
int a3[1][0] = { 1, 2 };
struct s0 b2[2][1] = { 1, 2 };
struct s2 r[2] = { 3, 4, 5 };
int tail[3] = { 4, 5 };
struct s2 q = { 1, 2 };
#include <stdio.h>
int main(void) {
  printf("%d %d %d\n", q.b, tail[0], tail[1]);
  return 0;
}
