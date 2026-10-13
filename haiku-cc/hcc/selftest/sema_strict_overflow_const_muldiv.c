// EXPECT-EXIT: 0
// A constant (x * c) / c overflows in the multiplication: the product wraps
// (it is not a UB rewrite candidate), so the quotient is 0 here.
int main(void) {
    if (1073741824 * 100 / 100 != 0) return 1;
    return 0;
}
