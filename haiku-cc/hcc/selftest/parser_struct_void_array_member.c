// EXPECT-ERROR: array of voids
// A struct member cannot be an array of void (C11 6.7.2.1).
struct A { int i; void x[1]; };
void foo(struct A a) {}
int main(void) { return 0; }
