// EXPECT-EXIT: 0
// A function returning a struct with a variably modified member: the caller's result
// block is allocated per statement, and an if-branch that returns must not leave a
// later statement's VLA chain pointing at an uninitialised slot.
void foo(int n) {
    struct S { int a[n]; };
    struct S mk(int k) {
        struct S s;
        s.a[0] = k;
        return s;
    }
    struct S rec(int d) {
        if (d == 0) return mk(1);
        return mk(d + 1);
    }
    struct S x;
    x = rec(0);
    if (x.a[0] != 1) __builtin_abort();
    x = rec(4);
    if (x.a[0] != 5) __builtin_abort();
    if (mk(7).a[0] != 7) __builtin_abort();
}
int main(void) { foo(3); return 0; }
