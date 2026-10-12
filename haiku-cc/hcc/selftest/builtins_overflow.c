// EXPECT-EXIT: 0
int main(void) {
    int r; unsigned u; long l; unsigned long ul; signed char sc;
    if (__builtin_mul_overflow(65536, 65536, &r) != 1) return 1;
    if (__builtin_add_overflow(2147483647, 1, &r) != 1 || r != -2147483647 - 1) return 2;
    if (__builtin_sub_overflow(0u, 1u, &u) != 1 || u != 4294967295u) return 3;
    if (__builtin_mul_overflow(-2, 3, &l) != 0 || l != -6) return 4;
    if (__builtin_mul_overflow(-1L, 1UL << 63, &ul) != 1) return 5;
    if (__builtin_add_overflow(100, 28, &sc) != 1 || sc != -128) return 6;
    return 0;
}
