// EXPECT-ERROR: too large
// A struct whose size exceeds PTRDIFF_MAX is rejected at its definition.
struct S { int a; char b[9223372036854775807L]; };
int main(void) { return 0; }
