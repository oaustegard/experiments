// EXPECT-ERROR: flexible array member in a struct with no named members
// An incomplete array member needs at least one other named member (C11 6.7.2.1p2).
struct S { int :1; int x[]; };
int main(void) { return 0; }
