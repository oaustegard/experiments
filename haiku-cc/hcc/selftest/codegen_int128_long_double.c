// EXPECT-EXIT: 0
typedef __int128 i128;
typedef unsigned __int128 u128;
__attribute__((noinline)) i128 to_i(long double x) { return (i128)x; }
__attribute__((noinline)) u128 to_u(long double x) { return (u128)x; }
int main(void) {
    i128 a = -((i128)1 << 100) + ((i128)1 << 40);
    long double la = (long double)a;
    if (to_i(la) != a) return 2;
    u128 u = ((u128)1 << 64) | ((u128)1 << 10);
    if (to_u((long double)u) != u) return 3;
    if (to_i(-3.9L) != -3) return 4;
    if (to_u(7.9L) != 7) return 5;
    return 0;
}
