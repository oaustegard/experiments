// EXPECT-EXIT: 0
// The values of calls, assignments and ++x are rvalues: qualifiers are dropped.
volatile int vi;
const int ci;
const int fci(void);
extern typeof (vi = 1) x1;
extern int x1;
extern typeof (++vi) x2;
extern int x2;
extern typeof (fci()) x3;
extern int x3;
int main(void) { return 0; }
