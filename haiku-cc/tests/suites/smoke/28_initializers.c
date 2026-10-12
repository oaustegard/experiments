int printf(const char *fmt, ...);
struct p { int a; char b[4]; int c; };
int main(void) { struct p v = { .c = 3, .a = 1, .b = "hi" }; int m[2][3] = { {1, 2, 3}, [1] = {4, 5} }; char s[] = "abc"; printf("%d %s %d %d %d %d\n", v.a, v.b, v.c, m[1][1], m[1][2], (int)sizeof s); return 0; }
