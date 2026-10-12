// EXPECT-EXIT: 112
// _Complex long double: x87 parts; * calls libgcc __mulxc3.
_Complex long double mul(_Complex long double a, _Complex long double b) { return a * b; }
int main(void) {
    _Complex long double a = 1.0L + 2.0iL, b = 3.0L - 4.0iL;
    _Complex long double c = mul(a, b);
    return (int)__real__ c * 10 + (int)__imag__ c;
}
