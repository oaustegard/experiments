int printf(const char *fmt, ...);
int twice(int x) { return 2 * x; }
int sq(int x) { return x * x; }
int apply(int (*f)(int), int v) { return f(v); }
int main(void) { int (*fs[2])(int) = { twice, sq }; printf("%d %d\n", apply(fs[0], 5), apply(fs[1], 5)); return 0; }
