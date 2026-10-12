// EXPECT-EXIT: 0
// EXPECT-STDOUT: first 10 20 1 500
// EXPECT-STDOUT: second 10 20 1 700
// setjmp on either side of a binary operator: no operand may be parked across the
// call, so the longjmp resumption sees the same left and right values gcc does.
#include <stdio.h>
#include <setjmp.h>
static int n;
static int nx(void) { return ++n; }
int main(void) {
    jmp_buf buf;
    volatile int i = 0;
    int r1 = 10 + setjmp(buf);
    int r2 = setjmp(buf) + 20;
    int r3 = nx() - setjmp(buf);
    int r4 = 100 * (5 + setjmp(buf));
    if (i < 1) {
        i++;
        printf("first %d %d %d %d\n", r1, r2, r3, r4);
        longjmp(buf, 2);
    }
    printf("second %d %d %d %d\n", r1, r2, r3, r4);
    return r4 == 700 ? 0 : 1;
}
