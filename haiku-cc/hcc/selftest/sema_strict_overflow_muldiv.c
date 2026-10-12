// EXPECT-EXIT: 0
// Signed (x * c) / c with constant c != 0 folds to x (overflow is UB);
// unsigned (x * c) / c must keep wrapping.
int sx(int x) { return (2 * x) / 2; }
int sy(int x) { return (x * -3) / -3; }
unsigned ux(unsigned x) { return (x * 2u) / 2u; }
int main(void) {
    if (sx(0x7fffffff) != 0x7fffffff) return 1;
    if (sy(5) != 5) return 2;
    if (ux(0x80000000u) != 0u) return 3;
    return 0;
}
