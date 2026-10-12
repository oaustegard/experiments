// EXPECT-EXIT: 0
// EXPECT-STDOUT: 1 1 1
// EXPECT-STDOUT: ffffffffffffffff fffffffffffffffb
// EXPECT-STDOUT: 20 3
// EXPECT-STDOUT: 1 1 10000 fffffffffffffffd
// EXPECT-STDOUT: 1
#include <stdio.h>
typedef __int128 s128; typedef unsigned __int128 u128;
struct S { int a; s128 x:100; u128 y:70; long long z; };
struct R { u128 p:90; s128 q:90; } __attribute__((scalar_storage_order("big-endian")));
int main(void) {
  struct S s = {1, 0, 0, 7};
  s.x = -5; s.y = ((u128)1 << 69) | 3;
  printf("%d %d %d\n", s.x == -5, s.x < 0, s.y == (((u128)1 << 69) | 3));
  printf("%llx %llx\n", (unsigned long long)(s.x >> 64), (unsigned long long)s.x);
  printf("%llx %llx\n", (unsigned long long)(s.y >> 64), (unsigned long long)s.y);
  struct R r; r.p = ((u128)1 << 80) + 9; r.q = -(s128)3;
  printf("%d %d %llx %llx\n", r.p == (((u128)1 << 80) + 9), r.q == -3, (unsigned long long)(r.p >> 64), (unsigned long long)(r.q));
  printf("%d\n", (s.x = 9) == 9);
  return 0;
}
