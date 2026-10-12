int printf(const char *fmt, ...);
enum color { RED, GREEN = 5, BLUE };
union u { int i; unsigned char c[4]; };
int main(void) { union u v; v.i = 0x01020304; printf("%d %d %d\n", BLUE, v.c[0], (int)sizeof(union u)); return 0; }
