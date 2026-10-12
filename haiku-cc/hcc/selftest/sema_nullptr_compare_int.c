// EXPECT-ERROR: [sema] invalid operands to binary ==
// EXPECT-FLAGS: -std=c23
// C23 nullptr is not an integer: comparing it with a non-null integer is invalid.
int f(void) { return (nullptr == 1); }
