// EXPECT-ERROR: doesn't match prototype
// An old-style definition whose parameter type differs from the prior
// prototype is rejected (the K&R float stays float before promotion).
int f(int a);
int f(a) long a; { return 0; }
int main(void) { return 0; }
