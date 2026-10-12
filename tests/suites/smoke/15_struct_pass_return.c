int printf(const char *fmt, ...);
struct s { long a; long b; long c; };
struct s mk(long x) { struct s r = { x, x * 2, x * 3 }; return r; }
long sum(struct s v) { return v.a + v.b + v.c; }
int main(void) { printf("%ld\n", sum(mk(7))); return 0; }
