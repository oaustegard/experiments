// EXPECT-EXIT: 0
// A call named alloca with a non-1 argument count (here an unprototyped
// declaration called with no arguments) is an ordinary call, not the alloca
// lowering, and must not crash codegen. It is never executed.
void *alloca ();
void *f(void) { return alloca (); }
void *alloca () { return 0; }
int main(void) { return f() != 0; }
