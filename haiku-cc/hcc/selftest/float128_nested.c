// EXPECT-EXIT: 0
// _Float128 libgcc calls nested inside operands that leave 8-byte pushes on the stack.
typedef _Float128 F;
double mix(double d, F f, int i, F g) { return d + (double)f + i + (double)g; }
int main(void) {
    F a = 3.0Q, b = 0.5Q;
    double d = 1.0 + (double)(a * b) * 2.0 + (double)(a - b);
    if (d != 1.0 + 3.0 + 2.5) return 1;
    if (mix(1.0, a * 2, 4, b / 2) != 1.0 + 6.0 + 4 + 0.25) return 2;
    int n = (int)(a + b) + (a > b) * 10;
    if (n != 13) return 3;
    long l = (long)(-a) * 2 + (long)((unsigned long)b * 0);
    if (l != -6) return 4;
    return 0;
}
