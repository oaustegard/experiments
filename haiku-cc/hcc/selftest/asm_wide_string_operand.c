// EXPECT-ERROR: a wide string is invalid in this context
// Wide string literals are rejected as asm operands and asm labels.
int f(void) { int x = 1; asm ("nop" : : : L"memory"); return x; }
