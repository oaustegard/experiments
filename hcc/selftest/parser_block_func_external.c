// EXPECT-ERROR: incompatible implicit declaration
// A block-scope function declaration has external linkage: a call to z() in
// another function conflicts with the implicit int z() and is an error.
void f(void) { long z(); }
void g(void) { z(); }
