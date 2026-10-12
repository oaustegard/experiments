// EXPECT-EXIT: 0
// Huge decimal exponents must compile quickly and saturate to inf / 0.
int main(void) {
    float a = 1e+100000000f;
    double b = 1e100000000;
    long double c = 1e-100000000l;
    if (a != 1.0f / 0.0f) return 1;
    if (b != 1.0 / 0.0) return 2;
    if (c != 0.0L) return 3;
    return 0;
}
