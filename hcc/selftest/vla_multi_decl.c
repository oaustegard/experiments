// EXPECT-EXIT: 0
// EXPECT-STDOUT: 1 -1 3 -2
#include <stdio.h>
void f(int n) {
    double u[n], v[n];
    int i;
    for (i = 0; i < n; i++) { u[i] = i; v[i] = -i; }
    printf("%g %g", u[1], v[1]);
    {
        int w[n + 1], x[2];
        w[n] = 3; x[1] = -2;
        printf(" %d %d\n", w[n], x[1]);
    }
}
int main(void) { f(4); return 0; }
