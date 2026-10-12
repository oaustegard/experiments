// EXPECT-ERROR: width not an integer constant
// A bit-field width naming a variable (even a const one) is not an integer constant expression.
void f(void) { const int m = 1; struct { int i : !!m; } x; (void)x; }
