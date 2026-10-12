// EXPECT-EXIT: 0
// __builtin_isnan/isinf/isfinite (and f/l/finite spellings) classify the operand's
// own bit pattern and return 0/1 without a libm call.
int main(void) {
    float finf = __builtin_inff(), fnan = __builtin_nanf("");
    double dninf = -__builtin_inf(), dnan = __builtin_nan(""), dsub = 1e-310;
    long double lnan = __builtin_nanl("");
    if (__builtin_isnan(fnan) != 1 || __builtin_isnan(finf) != 0) return 1;
    if (__builtin_isnan(lnan) != 1 || __builtin_isnanl(lnan) != 1) return 2;
    if (__builtin_isinf(dninf) != 1 || __builtin_isinff(finf) != 1) return 3;
    if (__builtin_isinf(dnan) != 0 || __builtin_isinfl(lnan) != 0) return 4;
    if (__builtin_isfinite(dsub) != 1 || __builtin_isfinite(dninf) != 0) return 5;
    if (__builtin_finite(dnan) != 0 || __builtin_finitef(fnan) != 0) return 6;
    if (__builtin_finitel(1.5L) != 1 || __builtin_isfinitef(finf) != 0) return 7;
    if (__builtin_isnan(3) != 0 || __builtin_isinf(7) != 0) return 8;
    return 0;
}
