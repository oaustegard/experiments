int printf(const char *fmt, ...);
int main(void) { int a[5]; for (int i = 0; i < 5; i++) a[i] = i * i; int s = 0; for (int i = 0; i < 5; i++) s += a[i]; printf("%d %d\n", s, a[3]); return 0; }
