// EXPECT-EXIT: 0
typedef __int128 V __attribute__((vector_size(32)));
typedef unsigned long long U64;
int main(void) {
    V a = { (__int128)-1LL, 0 };
    V b = { 1, 0 };
    V c = a + b;                       /* -1 + 1 = 0, carry out of low half */
    U64 *p = (U64 *)&c;
    if (p[0] != 0 || p[1] != 0) return 1;
    V d = b - a;                       /* 1 - (-1) = 2 */
    p = (U64 *)&d;
    if (p[0] != 2 || p[1] != 0) return 2;
    V e = -b;                          /* -1: all ones in both halves */
    p = (U64 *)&e;
    if (p[0] != ~0ULL || p[1] != ~0ULL) return 3;
    V f = a ^ b;                       /* -1 ^ 1 = -2 */
    p = (U64 *)&f;
    if (p[0] != ~1ULL || p[1] != ~0ULL) return 4;
    return 0;
}
