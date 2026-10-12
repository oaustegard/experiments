// EXPECT-EXIT: 0
// EXPECT-STDOUT: inf inf
// EXPECT-STDOUT: inf -inf
// EXPECT-STDOUT: -0.20 0.40
// EXPECT-STDOUT: nan nan
// EXPECT-STDOUT: 0.5 -0.5
// C99 Annex G: a nonzero complex quotient over a zero denominator is infinite,
// computed through libgcc's __divdc3 fallback when the inline formula gives NaN.
#include <stdio.h>
#include <complex.h>
int main(void) {
    volatile double z = 0.0;
    _Complex double a = 1.0 + 1.0 * I, b = z;
    _Complex double c = a / b;
    printf("%s %s\n", __builtin_isinf(creal(c)) ? "inf" : "?", __builtin_isinf(cimag(c)) ? "inf" : "?");
    c = (1.0 - 1.0 * I) / (0.0 - 0.0 * I);
    printf("%s %s\n", __builtin_isinf(creal(c)) ? "inf" : "?", __builtin_isinf(cimag(c)) && cimag(c) < 0 ? "-inf" : "?");
    c = (1.0 + 2.0 * I) / (3.0 - 4.0 * I);
    printf("%.2f %.2f\n", creal(c), cimag(c));
    c = (0.0 + 0.0 * I) / (0.0 + 0.0 * I);
    printf("%s %s\n", __builtin_isnan(creal(c)) ? "nan" : "?", __builtin_isnan(cimag(c)) ? "nan" : "?");
    c = ~(1.0 + 1.0 * I) / 2.0;
    printf("%g %g\n", creal(c), cimag(c));
    return 0;
}
