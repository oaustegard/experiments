// EXPECT-EXIT: 0
// GCC _FloatN spellings of inf/nan/libm builtins, the long double classifiers and
// integer-valued libm folding in a static initializer.
volatile _Float64x inf_cst = __builtin_inff64x();
static long lr = __builtin_lround(42.25);
static int il = __builtin_ilogb(1.0);
volatile long double neg = -__builtin_fabsf64x(-2.0f64x);
int main(void) {
    if (!__builtin_isinf(inf_cst) || __builtin_isinf((long double)1.0L)) return 1;
    if (!__builtin_isnan(__builtin_nanf64x("")) || __builtin_isnan(0.0L)) return 2;
    if (neg != -2.0L || lr != 42 || il != 0) return 3;
    return 0;
}
