// EXPECT-EXIT: 0
// GCC floating suffixes: w/W (__float80), q/Q (__float128), d/D (double),
// and the imaginary marker in either order (2.5fi, 2.5if).
double a = 0.5d, b = 0x1.8p1D, c = 0x1.p+1D;
long double l = 1.25w, m = 2.5W, n = 3.0L;
__float128 q = 1.0Q, r = 2.5q;
float f = 1.5f;
_Complex double ci = 0.5id, cj = 2.0fi;
int main(void) {
    if (a != 0.5 || b != 3.0 || c != 2.0 || f != 1.5f) return 1;
    if (l != 1.25L || m != 2.5L || n != 3.0L) return 2;
    if (q != 1.0Q || r != 2.5q) return 3;
    if (ci != 0.5i || cj != 2.0i) return 4;
    return 0;
}
