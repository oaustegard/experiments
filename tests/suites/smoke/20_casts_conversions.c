int printf(const char *fmt, ...);
int main(void) { int i = -1; unsigned u = i; long l = i; unsigned long ul = u; printf("%u %ld %lu %d\n", u, l, ul, (int)(char)300); return 0; }
