// EXPECT-ERROR: [parse]
// C11 6.2.2p7: a file-scope static declaration may not follow a non-static one.
int f(void);
static int f(void) { return 0; }
int main(void) { return 0; }
