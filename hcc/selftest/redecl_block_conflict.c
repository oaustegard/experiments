// EXPECT-ERROR: [parse]
// A second declaration of a block-scope object must have a compatible type.
int f(void) { int i; unsigned i[1]; return 0; }
int main(void) { return f(); }
