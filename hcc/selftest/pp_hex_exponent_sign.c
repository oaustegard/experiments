// EXPECT-EXIT: 0
// C99 and later: `+` after a hex 'p' belongs to the pp-number (0x1p+2 == 4.0).
// The C90 split (0x1p+f -> 0x1p, +, f) is covered by gcc-dg/cpp/pr23827_c90.c.
int main(void) {
    double d = 0x1p+2;
    double e = 0x1P-1;
    return (d == 4.0 && e == 0.5) ? 0 : 1;
}
