// EXPECT-EXIT: 0
// sizeof(*expr) of a pointer to a VLA evaluates expr, so the length set
// inside a statement expression is seen; a VLA in a statement expression
// does not hide the value of the expression.
extern void abort(void);
int f(void) {
    int n = 0;
    return sizeof(*({ n = 10; int x[n]; &x; }));
}
int main(void) {
    if (f() != 40) abort();
    if (sizeof(int[3]) != 12) abort();
    return 0;
}
