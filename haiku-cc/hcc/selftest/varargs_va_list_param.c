// EXPECT-EXIT: 0
// va_arg on a va_list parameter of a non-variadic function (no va_tmp yet)
// must not crash codegen; split struct {int; double} is read via the scratch.
#include <stdarg.h>
typedef struct { int i; double d; } QW;
static int take(int n, va_list ap) {
    QW q = va_arg(ap, QW);
    return n + q.i + (int)q.d;
}
static int wrap(int n, ...) {
    va_list ap;
    int r;
    va_start(ap, n);
    r = take(n, ap);
    va_end(ap);
    return r;
}
int main(void) {
    QW q = { 2, 3.0 };
    return wrap(0, q) == 5 ? 0 : 1;
}
