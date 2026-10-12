// EXPECT-EXIT: 0
// unsigned __int128 operands with a signed narrow or 128-bit result type
int main(void) {
    unsigned __int128 c = 1;
    if (__builtin_mul_overflow_p(59, -c, (short)0) != 1) return 1;
    if (__builtin_mul_overflow_p((unsigned __int128)3, (unsigned __int128)4, (signed char)0) != 0) return 2;
    if (__builtin_mul_overflow_p((unsigned __int128)1, (unsigned __int128)1, (__int128)0) != 0) return 3;
    if (__builtin_add_overflow_p((unsigned __int128)1 << 127, (unsigned __int128)0, (__int128)0) != 1) return 4;
    return 0;
}
