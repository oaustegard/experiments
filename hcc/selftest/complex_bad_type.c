// EXPECT-ERROR: cannot be applied
struct S { int a; };
_Complex struct S bad;
int main(void) { return 0; }
