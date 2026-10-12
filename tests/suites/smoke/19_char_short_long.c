int printf(const char *fmt, ...);
int main(void) { char c = 200; unsigned char uc = 200; short s = -40000; long l = 1L << 40; printf("%d %d %d %ld\n", c, uc, s, l); return 0; }
