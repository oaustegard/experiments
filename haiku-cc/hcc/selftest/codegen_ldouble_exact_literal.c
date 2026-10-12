// EXPECT-EXIT: 0
// Long double literals keep their exact x87 value: a magnitude beyond double range
// (1e4932L, 1e-4940L) must not fold to inf or zero, and a negated literal keeps its sign.
extern void abort(void);
static long double big = 1.18973149535723176502e+4932L;
static long double tiny = -3e-4000L;
int main(void) {
    volatile long double m = 1e4000L;
    volatile long double neg = -1e4000L;
    if (big <= 1e4000L) return 1;          /* finite, far beyond double */
    if (tiny >= 0 || tiny == 0) return 2;  /* sign and subnormal-range magnitude kept */
    if (m / 1e3000L < 1e999L) return 3;
    if (neg > 0 || neg == 0) return 4;
    return 0;
}
