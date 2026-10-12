// EXPECT-EXIT: 0
// __attribute__((cleanup)): the handler runs on block fallthrough, on break/continue
// out of the declaring loop body, and on return (keeping the returned value).
static int counter;
static void add(int *p) { counter += *p; }
static void bump(int *p) { counter += 100; }
static int early(int n) {
    int a __attribute__((cleanup(add))) = 5;
    if (n > 0) {
        int b __attribute__((cleanup(bump))) = 1;
        return 7 + n;
    }
    return 9;
}
static long double ld_ret(void) { int z __attribute__((cleanup(bump))) = 0; return 2.5L; }
int main(void) {
    int i, r;
    r = early(3);
    if (r != 10 || counter != 105) return 1;
    counter = 0;
    if (early(0) != 9 || counter != 5) return 2;
    counter = 0;
    if (ld_ret() != 2.5L || counter != 100) return 3;
    counter = 0;
    for (i = 0; i < 5; ++i) {
        int d __attribute__((cleanup(add))) = i;
        if (i == 2) break;
    }
    if (counter != 0 + 1 + 2 || i != 2) return 4;
    return 0;
}
