// EXPECT-EXIT: 0
// Only `(void *)0` is a null pointer constant here: the other arm (int *) decides the type.
int *p;
int *f(void) { return 1 ? (void *)0 : (int *)0; }
int *g(void) { return 1 ? (int *)0 : p; }
int main(void) { return f() != 0 || g() != 0; }
