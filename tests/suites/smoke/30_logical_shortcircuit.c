int printf(const char *fmt, ...);
int calls;
int t(void) { calls++; return 1; }
int f(void) { calls++; return 0; }
int main(void) { int r = (f() && t()) + (t() || f()) * 2 + !f() * 4; printf("%d %d\n", r, calls); return 0; }
