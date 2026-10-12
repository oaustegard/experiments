// EXPECT-ERROR: [sema] invalid operands to binary -
// Pointer difference requires compatible pointee types; _Atomic is not ignored.
int f(float *a, int *b) { return (int)(a - b); }
