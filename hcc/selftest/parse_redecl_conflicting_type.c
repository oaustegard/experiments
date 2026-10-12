// EXPECT-ERROR: [parse]
// A block-scope object may not be redeclared with a different type in the same scope (even after use).
int f(void) {
    int i;
    for (i = 0; i < 2; i++) ;
    float i;
    return 0;
}
