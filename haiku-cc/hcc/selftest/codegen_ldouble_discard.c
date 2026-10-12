// EXPECT-EXIT: 0
// A discarded long double value must pop the x87 stack: 20 discarded
// assignments in a loop would overflow the 8-slot stack and yield NaN.
extern void abort(void);
int main(void) {
    long double x = 1.0L;
    for (int i = 0; i < 20; i++) x *= 10.0L;
    if (!(x > 9.9e19L && x < 1.1e20L)) abort();
    long double y = 2.0L;
    y = 3.0L, y = 4.0L;
    long double z;
    z = 1.5L;
    if (z != 1.5L || y != 4.0L) abort();
    return 0;
}
