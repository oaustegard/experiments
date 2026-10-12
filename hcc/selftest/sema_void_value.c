// EXPECT-ERROR: void value not ignored
// A void call used as an operand of a binary operator is rejected.
void v(void);
int f(int a) { return a + v(); }
