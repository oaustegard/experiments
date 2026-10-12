// EXPECT-ERROR: flexible array member not at end of struct
// An incomplete array member must be the last member (C11 6.7.2.1p2).
struct S { int x[]; int y; };
int main(void) { return 0; }
