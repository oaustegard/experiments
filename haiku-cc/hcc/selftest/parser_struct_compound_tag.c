// EXPECT-EXIT: 0
// A compound literal that declares its tag is re-parsed on backtrack; not a redefinition.
int main(void) { return ((struct Q { int b; }){3}).b - 3; }
