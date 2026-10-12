// EXPECT-EXIT: 0
// x87 long double '>' and '>=' are false when an operand is NaN (unordered).
int main(void) {
    long double n = __builtin_nanl("");
    long double one = 1.0L;
    if (n > one) return 1;
    if (n >= one) return 2;
    if (one > n) return 3;
    if (one >= n) return 4;
    if (!(one > 0.5L)) return 5;
    if (!(one >= one)) return 6;
    return 0;
}
