// EXPECT-EXIT: 0
// __builtin_fma/nextafter take their libm prototype (double result, three operands).
int main(void) {
    double r = __builtin_fma(2.0, 3.0, 1.0);
    if (r != 7.0) return 1;
    float f = __builtin_fmaf(2.0f, 3.0f, 0.5f);
    if (f != 6.5f) return 2;
    if (__builtin_nextafter(1.0, 2.0) <= 1.0) return 3;
    return 0;
}
