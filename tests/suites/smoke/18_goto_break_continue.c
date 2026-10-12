int printf(const char *fmt, ...);
int main(void) { int s = 0; for (int i = 0; i < 100; i++) { if (i % 2) continue; if (i > 10) break; s += i; } if (s == 30) goto ok; return 1; ok: printf("%d\n", s); return 0; }
