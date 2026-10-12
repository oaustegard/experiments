// EXPECT-EXIT: 0
// va_arg with a pointer-to-VLA type evaluates the VLA bound (++i runs once).
#include <stdarg.h>
int i = 9;
int a[10];
static int *get(int n, ...) {
    va_list ap;
    int *p;
    va_start(ap, n);
    p = va_arg(ap, typeof(int (*)[++i]));
    va_end(ap);
    return p;
}
int main(void) {
    int *p = get(0, a);
    if (p != a) return 1;
    if (i != 10) return 2;
    return 0;
}
