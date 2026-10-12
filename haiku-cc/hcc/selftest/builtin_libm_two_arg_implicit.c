// EXPECT-EXIT: 0
// Implicit calls of two-operand libm builtins (fmod, fmax, fmin, hypot) take two arguments.
int main(void) {
    double d = 3.0;
    double a = fmax(d, 1.0);
    double b = fmin(d, 1.0);
    double c = fmod(d, 2.0);
    double e = hypot(d, 4.0);
    return (a == 3.0 && b == 1.0 && c == 1.0 && e == 5.0) ? 0 : 1;
}
