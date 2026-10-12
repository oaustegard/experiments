// EXPECT-EXIT: 0
// __builtin_complex is accepted in constant initializers (the real part is kept;
// _Complex itself is not modelled).
_Complex double c = __builtin_complex (0.0, 0.0);
static double d = __builtin_complex (1.5f, 2.0f);
int main(void) { return d == 1.5 ? 0 : 1; }
