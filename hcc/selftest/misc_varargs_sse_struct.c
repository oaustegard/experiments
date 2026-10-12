// EXPECT-EXIT: 0
// va_arg of a struct made of two doubles (SSE eightbytes) reads the xmm save area.
#include <stdarg.h>
typedef struct { double x, y; } point;
static int check(int n, ...) {
    va_list ap;
    int bad = 0;
    va_start(ap, n);
    for (int i = 0; i < n; i++) {
        point p = va_arg(ap, point);
        if (p.x != i + 1.0 || p.y != i + 2.0) bad = 1;
    }
    va_end(ap);
    return bad;
}
int main(void) {
    point a = {1.0, 2.0}, b = {2.0, 3.0};
    return check(2, a, b);
}
