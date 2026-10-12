// EXPECT-EXIT: 11
// Second argument of __builtin_expect must still be evaluated.
int f(int z) { if (__builtin_expect(0, z++)) return 7; return z; }
int main(void) { return f(10); }
