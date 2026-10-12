// EXPECT-EXIT: 0
// A float literal in a truth context is true/false at runtime, not folded as its integer part.
int main(void) {
    int x;
    if (0.1) x = 3; else x = 5;
    if (x != 3) return 1;
    if (0.5 ? 0 : 1) return 2;
    return 0;
}
