int printf(const char *fmt, ...);
#define SQR(x) ((x) * (x))
#define STR(x) #x
#define CAT(a, b) a##b
#if defined(SQR) && SQR(3) == 9
#define OK 1
#else
#define OK 0
#endif
int main(void) { int CAT(my, var) = SQR(4); printf("%d %s %d\n", myvar, STR(hi there), OK); return 0; }
