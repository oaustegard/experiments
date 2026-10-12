// EXPECT-EXIT: 0
// __builtin_complex(re, im) builds a _Complex value: constant at file scope and in
// static locals (parts folded, -0.0 kept), and a real object takes the real part.
_Complex double c = __builtin_complex (0.0, 0.0);
static double d = __builtin_complex (1.5f, 2.0f);
int main(void) {
    double a = 3.0;
    _Complex double r = __builtin_complex (a, -a);
    static _Complex float s = __builtin_complex ((float) 1, -0.0f);
    if (d != 1.5) return 1;
    if (__real__ r != 3.0 || __imag__ r != -3.0) return 2;
    if (__real__ c != 0.0 || __imag__ c != 0.0) return 3;
    if (__real__ s != 1.0f) return 4;
    if (__builtin_copysignf(1.0f, __imag__ s) > 0) return 5;
    return 0;
}
