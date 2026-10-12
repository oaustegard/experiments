// EXPECT-ERROR: [parse]
// Enumerator values must be integer constant expressions: a pointer cast is rejected.
enum e2 { E1 = (void *)4 };
int main(void) { return 0; }
