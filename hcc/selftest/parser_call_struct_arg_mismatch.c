// EXPECT-ERROR: [parse]
// Passing an int to a struct parameter is an incompatible argument (C11 6.5.2.2p7).
struct S { int a; };
void f(struct S);
void g(int x) { f(x); }
int main(void) { return 0; }
