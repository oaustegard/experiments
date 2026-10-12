int printf(const char *fmt, ...);
long f(long a, long b, long c, long d, long e, long f, long g, long h) { return a + 2*b + 3*c + 4*d + 5*e + 6*f + 7*g + 8*h; }
int main(void) { printf("%ld\n", f(1, 1, 1, 1, 1, 1, 1, 1)); return 0; }
