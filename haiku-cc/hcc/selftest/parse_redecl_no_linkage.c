// EXPECT-ERROR: [parse]
// A block-scope object (no linkage) may not be declared twice in the same scope, even with the same type.
int f(void) {
    int a;
    int a;
    return a;
}
