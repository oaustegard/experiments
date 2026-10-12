int printf(const char *fmt, ...);
struct bf { unsigned a : 3; unsigned b : 5; int c : 4; };
int main(void) { struct bf x = { 0 }; x.a = 9; x.b = 31; x.c = -3; printf("%u %u %d\n", x.a, x.b, x.c); return 0; }
