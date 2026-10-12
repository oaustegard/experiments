// EXPECT-EXIT: 0
int main(void) {
    unsigned char c = 1;
    if (__builtin_mul_overflow_p(4, (unsigned char)~c, 0)) return 1;
    if (!__builtin_mul_overflow_p(65536, 65536, 0)) return 2;
    if (!__builtin_mul_overflow_p(-1, 1UL << 63, 0UL)) return 3;
    if (__builtin_mul_overflow_p(-2, 3, (signed char)0)) return 4;
    return 0;
}
