// EXPECT-EXIT: 0
// GNU folding in static initializers: a const block-scope integer with a constant
// initializer, and a comparison of floating operands.
int main(void) {
    const int a = 3;
    static int b = a + 1;
    static int c = (float)0.5 == 0.5;
    static int d = 0.25 > 0.5;
    if (b != 4 || c != 1 || d != 0) return 1;
    return 0;
}
