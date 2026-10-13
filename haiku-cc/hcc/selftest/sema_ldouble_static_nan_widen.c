// EXPECT-EXIT: 0
// A file-scope long double (and _Float64x) initialized from a double NaN or inf
// must hold the x87 bit pattern of the converted value, not the double bits.
volatile _Float64x nanval = __builtin_nan("");
volatile long double nl = -__builtin_inf();
volatile _Float64x inf = __builtin_inf();
int main(void) {
    if (__builtin_isfinite(nanval) != 0) return 1;
    if (__builtin_isnan(nanval) == 0) return 2;
    if (__builtin_isinf(inf) == 0) return 3;
    if (__builtin_isinf(nl) == 0 || __builtin_signbit(nl) == 0) return 4;
    return 0;
}
