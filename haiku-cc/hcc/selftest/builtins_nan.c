// EXPECT-EXIT: 0
// __builtin_nan/nanf/nanl yield constant NaNs usable in static initializers.
extern void abort(void);
static const double d[] = { __builtin_nan(""), -__builtin_inf() };
static const float f = __builtin_nanf("");
int main(void) {
    if (d[0] == d[0]) abort();
    if (!(d[1] < 0)) abort();
    if (f == f) abort();
    return 0;
}
