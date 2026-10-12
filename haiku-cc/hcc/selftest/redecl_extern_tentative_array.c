// EXPECT-ERROR: [parse]
// A block-scope extern that completes a file-scope tentative `int x[];` to a
// size other than 1 (its implicit initializer size) is rejected.
int x[];
void f(void) {
    extern int x[2];
}
