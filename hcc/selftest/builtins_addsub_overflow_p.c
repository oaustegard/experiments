// EXPECT-EXIT: 0
int main(void) {
    signed char c = 127;
    unsigned char u = 0;
    if (!__builtin_add_overflow_p(c, 1, (signed char)0)) return 1;
    if (__builtin_add_overflow_p(c, 0, (signed char)0)) return 2;
    if (!__builtin_sub_overflow_p(u, 1, (unsigned char)0)) return 4;
    if (__builtin_sub_overflow_p(u, 0, (unsigned char)0)) return 5;
    if (!__builtin_sub_overflow_p(-2L, 0x7fffffffffffffffL, 0L)) return 6;
    if (__builtin_sub_overflow_p(5, 3, (int)0)) return 7;
    return 0;
}
