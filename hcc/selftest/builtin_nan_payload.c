// EXPECT-EXIT: 0
// __builtin_nans*(tag) is a signaling NaN (quiet bit clear) with the tag's payload
// (0 -> GCC's default bit below the quiet bit); __builtin_nan* sets the quiet bit.
// Static initializers keep the exact bits (no float round trip quiets them).
extern void abort(void);
static volatile float sf = __builtin_nansf("");
static volatile double sd = __builtin_nans("3");
union fb { float f; unsigned u; };
union db { double d; unsigned long long u; };
int main(void) {
    union fb a, b;
    union db c;
    a.f = __builtin_nanf("");
    b.f = __builtin_nansf("3");
    c.d = __builtin_nans("");
    if (a.u != 0x7fc00000u) abort();
    if (b.u != 0x7f800003u) abort();
    if (c.u != 0x7ff4000000000000ull) abort();
    union fb s;
    s.f = sf;
    if (s.u != 0x7fa00000u) abort();
    union db t;
    t.d = sd;
    if (t.u != 0x7ff0000000000003ull) abort();
    return 0;
}
