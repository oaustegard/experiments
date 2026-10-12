// EXPECT-EXIT: 0
// A user-declared function named alloca called with no arguments is an
// ordinary call, not the alloca builtin (which takes one argument).
void *alloca();
void *alloca() { return 0; }
void *f(void) { return alloca(); }
int main(void) { return f() != 0; }
