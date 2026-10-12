int printf(const char *fmt, ...);
int main(void) { int x = 1; int *p = &x; *p = 7; int **pp = &p; **pp += 1; printf("%d\n", x); return 0; }
