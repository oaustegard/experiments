// EXPECT-EXIT: 0
// long double -> unsigned long long must handle values >= 2^63 (fistp is
// signed-only); 1e19 is the value tcc's constant folding converts.
int main(void) {
    long double a = 1e19L;
    unsigned long long u = (unsigned long long)a;
    if (u != 10000000000000000000ULL) return 1;
    long double b = 9223372036854775808.0L;
    if ((unsigned long long)b != 9223372036854775808ULL) return 2;
    long double c = 5.0L;
    if ((unsigned long long)c != 5ULL) return 3;
    return 0;
}
