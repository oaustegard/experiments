// EXPECT-EXIT: 0
// EXPECT-STDOUT: 31.1 32.2 33.3
// Structs holding long double are passed in memory; a 16-aligned stack arg
// after a 24-byte stack arg must be laid out at its SysV offset (32, not 24).
#include <stdio.h>
struct h23 { double a, b, c; };
struct h32 { long double a, b; };
struct h31 { long double a; };
static void f(struct h31 x, struct h23 y, struct h32 z) {
    printf("%.1Lf %.1Lf %.1f\n", x.a, z.b, y.c);
}
int main(void) {
    struct h31 x = { 31.1 };
    struct h23 y = { 23.1, 23.2, 33.3 };
    struct h32 z = { 32.1, 32.2 };
    f(x, y, z);
    return 0;
}
