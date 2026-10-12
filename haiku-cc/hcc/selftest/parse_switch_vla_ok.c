// EXPECT-EXIT: 7
// Case labels inside a block that declares the VLA after the switch head are fine,
// and a VLA declared before the switch is not skipped by it.
int f(int a, int b) {
    int v[b];
    v[0] = 3;
    switch (a) {
    case 1: {
        int w[b];
        w[0] = 4;
        return w[0] + v[0];
    }
    default:
        switch (a) { case 2: return 9; }
        return 0;
    }
}
int main(void) { return f(1, 2); }
