// EXPECT-STDOUT: 0000000000000000 0000000000000005 0000000000000000
// EXPECT-EXIT: 0
#include <stdio.h>
typedef __int128 i128;
static i128 add3(long a, i128 x, long b, i128 y) { return x + y + a + b; }
static i128 neg(i128 v) { return -v; }
int main(void) {
  i128 r = add3(1, (i128)2, 1, (i128)1);
  i128 n = neg((i128)-7);
  printf("%016llx %016llx %016llx\n", (unsigned long long)(r >> 64), (unsigned long long)r, (unsigned long long)(n >> 64));
  return 0;
}
