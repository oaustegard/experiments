// EXPECT-EXIT: 0
int main(void) {
    __int128 m = (__int128)1 << 127;
    __int128 mn = -m, r;
    if (__builtin_mul_overflow(mn, (__int128)1, &r) || r != mn) return 1;
    if (!__builtin_mul_overflow(mn, (__int128)-1, &r)) return 2;
    if (__builtin_mul_overflow((__int128)-3, (__int128)4, &r) || r != -12) return 3;
    if (!__builtin_mul_overflow_p((__int128)1 << 100, (__int128)1 << 100, r)) return 4;
    unsigned __int128 u = ((unsigned __int128)1 << 64) + 1, ur;
    if (__builtin_mul_overflow(u, (unsigned __int128)3, &ur) || ur != 3 * u) return 5;
    if (__builtin_mul_overflow(u, u, &ur) != 1) return 6;
    return 0;
}
