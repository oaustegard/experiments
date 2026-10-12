// EXPECT-STDOUT: 00000010000000000000000000000000 7fffffffffffffffffffffffffffffff
// EXPECT-STDOUT: cmp 1 1 1 0
// EXPECT-STDOUT: div 00000000000000000000000000012345
// EXPECT-EXIT: 0
#include <stdio.h>
typedef unsigned __int128 u128;
typedef __int128 i128;
static void pr(u128 v) { printf("%016llx%016llx", (unsigned long long)(v >> 64), (unsigned long long)v); }
int main(void) {
  u128 a = (u128)1 << 100;
  u128 b = ((u128)1 << 64) | 12345;
  pr(a); printf(" "); pr(((u128)1 << 127) - 1); printf("\n");
  i128 s = -5;
  printf("cmp %d %d %d %d\n", s < 0, (u128)s > 5, s <= s, s >= 1);
  u128 q = (((u128)0x12345 << 64) | 0x6789abcdefULL) / (b | 1);
  printf("div "); pr(q); printf("\n");
  return 0;
}
