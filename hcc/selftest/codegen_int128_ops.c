// EXPECT-EXIT: 0
// EXPECT-STDOUT: mul 242babbd46e0f2da 45d63253c2365b03
// EXPECT-STDOUT: div 0 d403e0ba7df1
// EXPECT-STDOUT: cmp 0 0 1 1 0 1
// EXPECT-STDOUT: ucmp 1 1 0 0 0 1
// EXPECT-STDOUT: shr 0 7123456789a
// EXPECT-STDOUT: arg 7 -1
#include <stdio.h>
typedef __int128 s128;
typedef unsigned __int128 u128;
static void pr(const char *n, u128 v) { printf("%s %llx %llx\n", n, (unsigned long long)(v >> 64), (unsigned long long)v); }
static s128 neg(s128 v) { return -v; }
int main(void) {
    s128 a = ((s128)0x7123456789abcdefLL << 40) | 0x1234567;
    s128 b = -((s128)98765432123LL << 17) + 5;
    u128 ua = (u128)a, ub = (u128)b;
    pr("mul", (u128)(a * b));
    pr("div", (u128)(a / 12345) & 0xffffffffffffULL);
    printf("cmp %d %d %d %d %d %d\n", a < b, a <= b, a > b, a >= b, a == b, a != b);
    printf("ucmp %d %d %d %d %d %d\n", ua < ub, ua <= ub, ua > ub, ua >= ub, ua == ub, ua != ub);
    pr("shr", (u128)(ua >> 64) | 0x891a2b3c4d5e6f78ULL ? (ua >> 60) : 0);
    s128 n = neg(7);
    printf("arg %lld %lld\n", (long long)(a % 1000 ? 7 : 0), (long long)(n >> 1 == -4 ? -1 : 0));
    return 0;
}
