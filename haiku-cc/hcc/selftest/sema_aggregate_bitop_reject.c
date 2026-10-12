// EXPECT-ERROR: invalid operands to binary
// Bitwise '&' on a struct operand is a constraint violation (C11 6.5.10).
typedef struct { unsigned short s; } t;
int foo(t v) { return v & 42; }
int main(void) { return 0; }
