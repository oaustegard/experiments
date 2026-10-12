// EXPECT-ERROR: [sema]
// Shift operators require integer operands; a floating rhs is rejected.
int f(void) { return 1 << 0.1; }
