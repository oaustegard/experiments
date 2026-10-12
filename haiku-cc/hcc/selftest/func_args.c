// EXPECT-EXIT: 0
int add6(int a, int b, int c, int d, int e, int f) { return a + b + c + d + e + f; }
long add8(long a, long b, long c, long d, long e, long f, long g, long h) {
    return a + b + c + d + e + f + g * 10 + h * 100;
}
static int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }
int main(void) {
    if (add6(1, 2, 3, 4, 5, 6) != 21) return 1;
    if (add8(1, 1, 1, 1, 1, 1, 2, 3) != 326) return 2;
    if (fact(5) != 120) return 3;
    return 0;
}
