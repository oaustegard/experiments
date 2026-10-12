// EXPECT-EXIT: 0
// Constant float->int conversions saturate to the target range (GCC folding).
int f1(void) { return (int)2147483648.0f; }
int f2(void) { return (int)(float)2147483647; }
int g(void) { return (unsigned char)300.0; }
int main(void) {
    if (f1() != 2147483647) return 1;
    if (f2() != 2147483647) return 2;
    if ((int)-1e30 != -2147483647 - 1) return 3;
    if (g() != 255) return 4;
    return 0;
}
