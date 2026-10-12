// EXPECT-EXIT: 0
// Hex floating constant out of range for double must not crash the lexer;
// it becomes infinity (with a warning), so the comparison below holds.
int main(void) {
    double x = 0x1.p+2000;
    return x > 1.0e308 ? 0 : 1;
}
