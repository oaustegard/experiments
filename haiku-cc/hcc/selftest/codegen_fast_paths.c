// EXPECT-EXIT: 0
// Fast paths: constant right operands, direct local loads, zeroing small
// locals, and compare-then-branch fusion must keep C semantics.
int main(void) {
    signed char c = -3;
    unsigned short s = 65535;
    int i = 7;
    long l = -5;
    int zero[5] = {0};
    if (c * 2 != -6) return 1;
    if (s + 1 != 65536) return 2;
    if (i - (-1) != 8) return 3;
    if (!(l < 0)) return 4;
    if ((c < 0) == 0) return 5;
    if (l != -5 || (i >= 7 && 0)) return 6;
    for (int k = 0; k < 5; k++) if (zero[k]) return 7;
    return 0;
}
