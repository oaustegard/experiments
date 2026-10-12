// EXPECT-EXIT: 0
// EXPECT-STDOUT: 10 14 1
// Sibling blocks may share frame slots; values in the enclosing scope and
// statement-expression results must stay intact.
#include <stdio.h>

struct P { int a, b; };

static int pick(int c) {
    int r = 0;
    if (c) { long big[200]; big[0] = 5; r += (int)big[0]; }
    else { char buf[3000]; buf[0] = 7; r += buf[0]; }
    { double d[50]; d[1] = 2.5; r += (int)(d[1] * 2); }
    return r;
}

static int sum2(struct P x, struct P y) { return x.a + y.b; }

int main(void) {
    printf("%d %d ", pick(1), pick(0) + 2);
    int s = sum2(({ struct P p = {1, 2}; p; }), ({ struct P q = {3, 4}; q; }));
    printf("%d\n", s == 5 ? 1 : 0);
    return 0;
}
