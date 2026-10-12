// EXPECT-EXIT: 0
// __builtin_fpclassify / __builtin_isnormal / __builtin_isinf_sign classify the
// raw bits of float, double, _Float16 and long double (glibc <math.h> macros).
#define FP_NAN 0
#define FP_INFINITE 1
#define FP_ZERO 2
#define FP_SUBNORMAL 3
#define FP_NORMAL 4
int main(void) {
    volatile double d = 1e-310;
    volatile float f = 1e-40f;
    volatile _Float16 h = (_Float16)1e-6;
    volatile long double l = 1.0L;
    volatile double nz = -0.0;
    if (__builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, d) != FP_SUBNORMAL) return 1;
    if (__builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, f) != FP_SUBNORMAL) return 2;
    if (__builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, h) != FP_SUBNORMAL) return 3;
    if (__builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, l) != FP_NORMAL) return 4;
    if (__builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, nz) != FP_ZERO) return 5;
    if (__builtin_isnormal(d) || !__builtin_isnormal(l) || __builtin_isnormal(f)) return 6;
    if (__builtin_isinf_sign(-1.0 / (double)0.0 * 1.0) != -1) return 7;
    if (__builtin_isinf_sign(f) != 0) return 8;
    if (__builtin_isinf_sign(l * 0.0L + 1.0L / 0.0L) != 1) return 9;
    if (__builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, (double)(0.0 / 0.0)) != FP_NAN) return 10;
    return 0;
}
