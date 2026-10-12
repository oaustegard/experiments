int printf(const char *fmt, ...);
int len(const char *s) { int n = 0; while (*s++) n++; return n; }
int main(void) { const char *s = "abcdef"; printf("%d %c %s\n", len(s), s[2], s + 3); return 0; }
