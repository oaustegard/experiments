// EXPECT-EXIT: 0
// Compound assignment to a complex lvalue whose address has side effects (evaluated once).
double _Complex g[2];
int calls;
double _Complex *pick(void) { calls++; return &g[1]; }
int main(void) {
    g[1] = 1.0 + 2.0 * 1i;
    pick()[0] += 1;
    if (calls != 1) return 1;
    if (__real__ g[1] != 2.0 || __imag__ g[1] != 0.0 + 2.0) return 2;
    return 0;
}
