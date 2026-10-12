// EXPECT-EXIT: 0
// __builtin_tgmath picks the candidate whose parameter type is the value's exact
// type: _Float32 is distinct from float, _Float32x from double.
float tf(float x) { return x + 1; }
_Float32 tf32(_Float32 x) { return x + 4; }
_Float32x tf32x(_Float32x x) { return x + 8; }
double td(double x) { return x + 2; }
#define T(x) __builtin_tgmath(tf, tf32, tf32x, td, x)
int main(void) {
    _Float32 a = 1;
    _Float32x b = 1;
    int i = 2;
    if (T(a) != 5) return 1;
    if (T(b) != 9) return 2;
    if (T(i) != 4) return 3;
    if (T(1.0f) != 2) return 4;
    return 0;
}
