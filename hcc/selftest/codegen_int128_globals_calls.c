// EXPECT-EXIT: 0
// EXPECT-STDOUT: 1234 5678
// EXPECT-STDOUT: fffffff000000000 0
// EXPECT-STDOUT: 4000000 0 -7
// EXPECT-STDOUT: 3 140
// EXPECT-STDOUT: 2 1
// EXPECT-STDOUT: 7 0 1
// EXPECT-STDOUT: 4 0
// EXPECT-STDOUT: 3000000000 1
// EXPECT-STDOUT: 80 6
// EXPECT-STDOUT: 1 1 1
// EXPECT-STDOUT: 1
// EXPECT-STDOUT: -1 1
#include <stdio.h>
typedef __int128 s128; typedef unsigned __int128 u128;
static u128 G = ((u128)0x1234 << 64) | 0x5678;
s128 NEG = -((s128)1 << 100);
s128 arr[2] = {(s128)1 << 90, -7};
struct P { int k; u128 v; } gp = {3, (u128)5 << 70};
u128 f(u128 x, int k, s128 y) { return x * k + (u128)y; }
s128 g(s128 a, s128 b, s128 c, s128 d, s128 e, s128 f2, s128 h) { return a - b + c * d - e + f2 * h; }
int main(void) {
  printf("%llx %llx\n", (unsigned long long)(G >> 64), (unsigned long long)G);
  printf("%llx %llx\n", (unsigned long long)(NEG >> 64), (unsigned long long)(NEG));
  printf("%llx %llx %lld\n", (unsigned long long)(arr[0] >> 64), (unsigned long long)arr[0], (long long)arr[1]);
  printf("%d %llx\n", gp.k, (unsigned long long)(gp.v >> 64));
  s128 i = (s128)1 << 65; i++; ++i; i--; printf("%llx %llx\n", (unsigned long long)(i >> 64), (unsigned long long)i);
  s128 t = i ? 7 : 8; printf("%lld %d %d\n", (long long)t, (i && 0), (i || 0));
  i <<= 2; i >>= 1; i |= 1; i ^= 3; i &= ~(s128)0 >> 1; printf("%llx %llx\n", (unsigned long long)(i >> 64), (unsigned long long)i);
  u128 r = f(((u128)1 << 100) + 1, 3, -2); printf("%llx %llx\n", (unsigned long long)(r >> 64), (unsigned long long)r);
  s128 q = g(1, 2, 3, 4, 5, (s128)1 << 70, 2); printf("%llx %llx\n", (unsigned long long)(q >> 64), (unsigned long long)q);
  int n = 3; long long ll = -2; printf("%d %d %d\n", (s128)n > ll, (u128)ll > (u128)n, (s128)ll < n);
  printf("%d\n", (int)((s128)1 << 70 >> 70));
  s128 m = 0; m = m ? m : (s128)-1; printf("%lld %d\n", (long long)m, m == -1);
  return 0;
}
