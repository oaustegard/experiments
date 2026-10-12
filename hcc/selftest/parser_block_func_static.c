// EXPECT-ERROR: [parse]
// A block-scope function declaration may not have the static storage class.
int f(void) {
    static int g(void);
    return g();
}
