// EXPECT-ERROR: [codegen]
// A call returning a struct with a variably modified member inside an operand that
// is already pushed (here a statement-expression argument) is not supported yet.
int g(int x, int y) { return x + y; }
void foo(int n) {
    struct S { int a[n]; };
    struct S mk(int k) { struct S s; s.a[0] = k; return s; }
    int r = g(({ int z = mk(7).a[0]; z; }), 0);
    (void)r;
}
int main(void) { foo(3); return 0; }
